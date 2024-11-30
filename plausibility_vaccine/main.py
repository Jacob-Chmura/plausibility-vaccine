import logging
import pathlib
from typing import Dict, Optional

import evaluate
import numpy as np
from adapters import AdapterArguments, AdapterTrainer, AutoAdapterModel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from plausibility_vaccine.adapters import setup_adapters
from plausibility_vaccine.data import get_data, preprocess_function
from plausibility_vaccine.util.args import (
    AdapterArguments,
    DataTrainingArguments,
    ModelArguments,
    parse_args,
)
from plausibility_vaccine.util.logging import setup_basic_logging
from plausibility_vaccine.util.seed import seed_everything


def get_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    output_dir = pathlib.Path(training_args.output_dir)
    last_checkpoint = None
    if output_dir.is_dir() and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(list(output_dir.iterdir())) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logging.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )

    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    return checkpoint


def run(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    adapter_args: AdapterArguments,
) -> None:
    logging.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, '
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logging.debug(f'Training/evaluation parameters {training_args}')

    raw_datasets, label_list = get_data(data_args, cache_dir=model_args.cache_dir)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.pretrained_model_name,
        num_labels=len(label_list),
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

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    with training_args.main_process_first(desc='dataset map pre-processing'):
        raw_datasets = raw_datasets.map(
            lambda batch: preprocess_function(batch, tokenizer, label_to_id),
            batched=True,
            desc='Running tokenizer on dataset',
        )

    # Evaluation Metrics
    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        metrics = ['accuracy']
        logits, labels = p
        preds = np.argmax(logits, axis=1)

        result = {}
        for metric in metrics:
            metric_obj = evaluate.load(metric)
            result.update(metric_obj.compute(predictions=preds, references=labels))
        return result

    # Setup adapters
    model = setup_adapters(model, adapter_args, data_args.task_name, label_list)

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

    logging.info('*** Training ***')
    train_result = trainer.train(resume_from_checkpoint=get_checkpoint(training_args))
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)

    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    logging.info('*** Evaluate ***')
    metrics = trainer.evaluate(predict_dataset)
    metrics['eval_samples'] = len(predict_dataset)
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)


def main() -> None:
    config_yaml = 'config/base.yaml'
    model_args, data_args, training_args, adapter_args = parse_args(config_yaml)
    setup_basic_logging()
    seed_everything(training_args.seed)
    run(model_args, data_args, training_args, adapter_args)


if __name__ == '__main__':
    main()
