from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics
from typing import Any, Optional, Union
import wandb

class NERTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]]=None,
            trial=None,
            ignore_keys_for_eval: Optional[list[str]]=None,
            **kwargs,
        ):
        super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss for the model.

        Args:
            model: Model for token classification.
            inputs: Inputs to the model.
            return_outputs: Whether to return the outputs.

        Returns:
            Loss and optionally the outputs of the model.
        """
        # Unpack inputs
        # labels = inputs.pop("labels")
        
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        # Forward pass
        outputs = model(**inputs)
        
        # Compute loss
        loss = outputs.loss
        
        if return_outputs:
            return (loss, outputs)
        
        return loss


def create_training_arguments(config) -> TrainingArguments:
    """
    Create and return the training arguments for the model.

    Returns:
        Training arguments for the model.

    NOTE: You can change the training arguments as needed.
    # Below is an example of how to create training arguments. You are free to change this.
    # ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        push_to_hub=False,
        eval_strategy="epoch",
        save_strategy="epoch",  # Save the model every epoch
        # eval_steps=1,
        learning_rate=config.get('learning_rate', 5e-5),
        lr_scheduler_type=config.get('lr_scheduler_type', 'linear'),
        per_device_train_batch_size=config.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=8,
        num_train_epochs=config.get('epochs', 10),
        weight_decay=config.get('weight_decay', 0),
        # max_grad_norm=config.get('max_grad_norm', 0.99),
        save_total_limit=2,
        run_name=wandb.run.name,
        report_to="wandb",
        logging_dir="logs",
        auto_find_batch_size=True,
        torch_empty_cache_steps=1,
    )
    
    return training_args


def build_trainer(model, tokenizer, tokenized_datasets, config) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

    Args:
        model: Model for token classification.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.
    """
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args: TrainingArguments = create_training_arguments(config)

    return NERTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer
    )
