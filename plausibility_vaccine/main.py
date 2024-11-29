import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import datasets
import evaluate
import numpy as np
import transformers
from adapters import (
    AdapterArguments,
    AdapterConfig,
    AdapterTrainer,
    AutoAdapterModel,
    setup_adapter_training,
)
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from plausibility_vaccine.util import seed_everything, setup_basic_logging

seed = 0


def load_pretrained_model(
    model_name: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    logging.info('Loading AutoTokenizer for: %s', model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logging.info('Loading AutoModel for: %s', model_name)
    model = AutoAdapterModel.from_pretrained(model_name)

    return model, tokenizer


def add_adapter_head(
    model: PreTrainedModel,
    adapter_name: str,
    adapter_config: Union[str, AdapterConfig],
) -> PreTrainedModel:
    model.add_adapter(adapter_name, config=adapter_config)
    return model


def add_plausibility_adapter_head(model: PreTrainedModel) -> PreTrainedModel:
    model.add_classification_head('plausibility', num_labels=2)
    return model


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    task_name: str = field(
        metadata={'help': 'The name of the task to train on'},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={'help': 'Overwrite the cached preprocessed datasets or not.'},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={'help': 'A csv or a json file containing the training data.'},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={'help': 'A csv or a json file containing the validation data.'},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={'help': 'A csv or a json file containing the test data.'},
    )


@dataclass
class ModelArguments:
    pretrained_model_name: str = field(
        default='foo',
        metadata={
            'help': 'Path to pretrained model or model identifier from huggingface.co/models'
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'
        },
    )


def run(config_yaml: Union[str, pathlib.Path]) -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments)
    )

    model_args, data_args, training_args, adapter_args = parser.parse_yaml_file(
        config_yaml
    )

    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, '
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.debug(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Loading a dataset from your local files.
    data_files = {
        'train': data_args.train_file,
        'test': data_args.test_file,
    }
    for key in data_files.keys():
        logger.info(f'load a local file for {key}: {data_files[key]}')

    # Loading a dataset from local csv files
    raw_datasets = load_dataset(
        'csv', data_files=data_files, cache_dir=model_args.cache_dir
    )

    label_list = raw_datasets['train'].unique('label')
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.pretrained_model_name,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name,
        cache_dir=model_args.cache_dir,
    )
    model = AutoAdapterModel.from_pretrained(
        model_args.pretrained_model_name,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Add head
    model.add_classification_head(
        data_args.task_name,
        num_labels=num_labels,
        id2label={i: v for i, v in enumerate(label_list)},
    )
    model.add_adapter(data_args.task_name, config='seq_bn')
    model.set_active_adapters(data_args.task_name)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    def preprocess_function(examples: Dict[str, List]) -> Dict[str, List]:
        # Tokenize the texts
        args = examples['subject'], examples['verb'], examples['object']
        result = tokenizer(*args, padding='max_length', max_length=8, truncation=True)

        # Map labels to IDs
        if label_to_id is not None and 'label' in examples:
            result['label'] = [label_to_id[l] for l in examples['label']]
        return result

    with training_args.main_process_first(desc='dataset map pre-processing'):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc='Running tokenizer on dataset',
        )

    # Get the metric function
    metric = evaluate.load('accuracy', cache_dir=model_args.cache_dir)

    # You can define your custom compute_metrics function.
    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result['combined_score'] = np.mean(list(result.values())).item()
        return result

    # Setup adapters
    setup_adapter_training(model, adapter_args, data_args.task_name)

    # Get datasets
    train_dataset, predict_dataset = raw_datasets['train'], raw_datasets['test']

    # Initialize our Trainer
    model.train_adapter(data_args.task_name)
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
    )

    # Training
    logger.info('*** Training ***')
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    logger.info('*** Predict ***')
    # Removing the `label` columns because it contains -1 and Trainer won't like that.
    predict_dataset = predict_dataset.remove_columns('label')
    predictions = trainer.predict(
        predict_dataset, metric_key_prefix='predict'
    ).predictions
    predictions = np.argmax(predictions, axis=1)

    output_predict_file = os.path.join(
        training_args.output_dir, f'predict_results_{data_args.task_name}.txt'
    )
    if trainer.is_world_process_zero():
        with open(output_predict_file, 'w') as writer:
            logger.info(f'***** Predict results {data_args.task_name} *****')
            writer.write('index\tprediction\n')
            for index, item in enumerate(predictions):
                item = label_list[item]
                writer.write(f'{index}\t{item}\n')


def main() -> None:
    setup_basic_logging()
    seed_everything(seed)
    run('config/base.yaml')


if __name__ == '__main__':
    main()
