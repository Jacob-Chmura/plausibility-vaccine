import logging
import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import evaluate
import transformers
from adapters import (
    AdapterArguments,
    AdapterTrainer,
    AutoAdapterModel,
    setup_adapter_training,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


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


def run(config_yaml: pathlib.Path):
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
    logger.info(f'Training/evaluation parameters {training_args}')

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
        #'train': data_args.train_file,
        'validation': data_args.validation_file,
        'test': data_args.test_file,
    }
    for key in data_files.keys():
        logger.info(f'load a local file for {key}: {data_files[key]}')

    # Loading a dataset from local csv files
    raw_datasets = load_dataset(
        'csv',
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        delimiter='\t',
        column_names=['label', 's', 'v', 'o', 's-sense', 'o-sense'],
    )

    label_list = raw_datasets['validation'].unique('label')
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

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    def preprocess_function(examples):
        # Tokenize the texts
        print(examples.keys())
        exit()
        args = examples[sentence1_key], examples[sentence2_key]
        result = tokenizer(*args)

        # Map labels to IDs
        if label_to_id is not None and 'label' in examples:
            result['label'] = [
                (label_to_id[l] if l != -1 else -1) for l in examples['label']
            ]
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

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result['combined_score'] = np.mean(list(result.values())).item()
        return result

    # Setup adapters
    setup_adapter_training(model, adapter_args, data_args.task_name)

    # Initialize our Trainer
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
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

    # Evaluation
    logger.info('*** Evaluate ***')
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]

    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics['eval_samples'] = len(eval_dataset)

        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    logger.info('*** Predict ***')
    tasks = [data_args.task_name]
    predict_datasets = [predict_dataset]

    for predict_dataset, task in zip(predict_datasets, tasks):
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = predict_dataset.remove_columns('label')
        predictions = trainer.predict(
            predict_dataset, metric_key_prefix='predict'
        ).predictions
        predictions = np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(
            training_args.output_dir, f'predict_results_{task}.txt'
        )
        if trainer.is_world_process_zero():
            with open(output_predict_file, 'w') as writer:
                logger.info(f'***** Predict results {task} *****')
                writer.write('index\tprediction\n')
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f'{index}\t{item}\n')
