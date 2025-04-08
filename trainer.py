from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics
from typing import Any, Optional, Union

class NERTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_run = None
        if self.args.wandb:
            self.init_wandb()
    
    def log(self, logs: dict, *args, **kwargs):
        if self.wandb_run is not None:
            self.wandb_run.log(logs)
        super().log(logs, *args, **kwargs)

    def init_wandb(self):
        import wandb
        self.wandb_run = wandb.init(
            project="NER",
            name=self.args.run_name,
            config=self.args.to_dict(),
            dir=self.args.output_dir,
            reinit=True,
        )

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]]=None,
            trial=None,
            ignore_keys_for_eval: Optional[list[str]]=None,
            **kwargs,
        ):
        if self.wandb_run is not None:
            self.wandb_run.watch(self.model, log="all", log_graph=True)
        super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def compute_loss(self, model, inputs, return_outputs=False):
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
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        
        # Compute loss
        loss = outputs.loss
        
        if return_outputs:
            return (loss, outputs)
        
        return loss


def create_training_arguments() -> TrainingArguments:
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
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=12,
        weight_decay=0.01,
        save_total_limit=2
    )
    
    return training_args


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
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

    training_args: TrainingArguments = create_training_arguments()

    return NERTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        wandb=True,
        run_name="NER-Training"
    )
