import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import evaluate
import numpy as np
from adapters import AdapterTrainer, AutoAdapterModel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from plausibility_vaccine.data import get_data, preprocess_function
from plausibility_vaccine.fine_tune import setup_adapters
from plausibility_vaccine.util.args import (
    FinetuningArgument,
    FinetuningArguments,
    ModelArguments,
)


def run(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    finetuning_args: FinetuningArguments,
) -> None:
    logging.info(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, '
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logging.debug(f'Training/evaluation parameters {training_args}')

    for _, task_args in finetuning_args.pretraining_tasks.items():
        _run_task(model_args, training_args, task_args)

    for _, task_args in finetuning_args.downstream_tasks.items():
        _run_task(model_args, training_args, task_args)


def _run_task(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    task_args: FinetuningArgument,
) -> None:
    logging.info('Setting up pre-training for task: %s', task_args.data_args.task_name)
    data_args, adapter_args = task_args.data_args, task_args.adapter_args
    raw_datasets, label_list = get_data(data_args)

    model, tokenizer = _load_pretrained_model(model_args, label_list=label_list)

    with training_args.main_process_first(desc='dataset map pre-processing'):
        raw_datasets = raw_datasets.map(
            lambda batch: preprocess_function(batch, tokenizer, label_list),
            batched=True,
            desc='Running tokenizer on dataset',
        )

    # Evaluation Metrics
    if data_args.is_regression:
        metrics = ['accuracy']
    else:
        metrics = ['mse']

    logging.info('Using evaluation metrics: %s', metrics)

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        logits, labels = p

        if data_args.is_regression:
            preds = np.squeeze(logits)
        else:
            preds = np.argmax(logits, axis=1)

        result = {}
        for metric in metrics:
            metric_obj = evaluate.load(metric)
            result.update(metric_obj.compute(predictions=preds, references=labels))
        return result

    # Setup adapters
    model = setup_adapters(model, adapter_args, data_args.task_name, label_list)

    # Initialize our Trainer
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets['train'],
        eval_dataset=raw_datasets.get('test'),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
    )
    _run_trainer(trainer, training_args)


def _load_pretrained_model(
    model_args: ModelArguments, label_list: Optional[List[str]]
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    logging.info(f'Loading pre-trained model: {model_args}, label_list: {label_list}')

    config = AutoConfig.from_pretrained(
        model_args.pretrained_model_name,
        num_labels=len(label_list) if label_list is not None else 1,
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
    if label_list is not None:
        label_to_id = {v: i for i, v in enumerate(label_list)}
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label for label, id in model.config.label2id.items()
        }

    logging.debug('Loaded pre-trained model: %s', model)
    logging.debug('Loaded pre-trained tokenizer: %s', tokenizer)
    return model, tokenizer


def _get_checkpoint(training_args: TrainingArguments) -> Optional[str]:
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


def _run_trainer(trainer: AdapterTrainer, training_args: TrainingArguments) -> None:
    logging.info('*** Training ***')
    train_result = trainer.train(resume_from_checkpoint=_get_checkpoint(training_args))
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    logging.info('*** Evaluate ***')
    metrics = trainer.evaluate()
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)
