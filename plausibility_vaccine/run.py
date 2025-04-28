import copy
import logging
from pathlib import Path
from typing import Dict, Union

import evaluate
import numpy as np
from adapters import AdapterTrainer
from scipy.special import softmax
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from plausibility_vaccine.adapter import setup_adapters
from plausibility_vaccine.data import get_data, preprocess_function
from plausibility_vaccine.fine_tune import load_pretrained_model
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

    for task_name, task_args in finetuning_args.pretraining_tasks.items():
        logging.info('Running %s', task_name)
        _run_task(model_args, training_args, task_args, downstream=False)

    for task_name, task_args in finetuning_args.downstream_tasks.items():
        logging.info('Running %s', task_name)
        _run_task(model_args, training_args, task_args, downstream=True)


def _run_task(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    task_args: FinetuningArgument,
    downstream: bool,
) -> None:
    logging.info('Setting up pre-training for task: %s', task_args.data_args.task_name)
    data_args, adapter_args = task_args.data_args, task_args.adapter_args
    raw_datasets, label_list = get_data(data_args)

    model, tokenizer = load_pretrained_model(
        model_args,
        label_list=label_list,
        use_adapter_for_task=task_args.use_adapter_for_task,
    )

    with training_args.main_process_first(desc='dataset map pre-processing'):
        for split in ['train', 'test']:
            raw_datasets[split] = raw_datasets[split].map(
                lambda batch: preprocess_function(batch, tokenizer, label_list),
                batched=True,
                desc=f'Running tokenizer on {split} dataset',
            )

    # Evaluation Metrics
    if data_args.is_regression:
        metrics = ['mse']
    else:
        metrics = ['accuracy', 'precision', 'recall', 'f1']

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
            kwargs = {'predictions': preds, 'references': labels}
            if metric in ['precision', 'recall', 'f1']:
                kwargs['average'] = 'macro'
            result.update(metric_obj.compute(**kwargs))
        return result

    # Setup adapters
    model = setup_adapters(
        model,
        adapter_args,
        data_args.task_name,
        label_list,
        task_args.fusion,
        task_args.use_adapter_for_task,
        training_args.output_dir,
    )

    # Initialize our Trainer
    training_args_ = copy.deepcopy(training_args)
    training_args_.output_dir += f'/{data_args.task_name}'

    if task_args.use_adapter_for_task:
        trainer_class = AdapterTrainer
    else:
        trainer_class = Trainer
    trainer = trainer_class(
        model=model,
        args=training_args_,
        train_dataset=raw_datasets['train'],
        eval_dataset=raw_datasets['test'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
    )
    _run_trainer(trainer, downstream)

    # Deactivate the trained adapters to restore the base model
    if task_args.use_adapter_for_task:
        logging.info(f'Deleting adapter {data_args.task_name} from base model')
        model.delete_adapter(data_args.task_name)


def _run_trainer(trainer: Union[AdapterTrainer, Trainer], downstream: bool) -> None:
    logging.info('*** Training ***')
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    logging.info('*** Evaluate ***')
    metrics = trainer.evaluate()
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)

    if downstream:
        logging.info('*** Generating predictions on training set. ***')
        predictions_output = trainer.predict(trainer.train_dataset)
        logits = predictions_output.predictions
        label_ids = predictions_output.label_ids
        probabilites = softmax(logits, axis=1)
        binary_predictions = np.argmax(probabilites, axis=1)

        output_dir = Path(trainer.args.output_dir)
        predictions_file = output_dir / 'train_binary_predictions.csv'

        np.savetxt(
            predictions_file,
            np.column_stack((binary_predictions, label_ids)),
            delimiter=',',
            header='predicted_label, true_label',
            fmt='%d',
            comments='',
        )
        logging.info(f'Train predictions saved to {predictions_file}')
