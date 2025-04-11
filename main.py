import os

from dataset import build_dataset, preprocess_data
from model import initialize_model
from tokenizer import initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds

import wandb

# os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb_args = {
    'project': 'ner',
    'name': 'ner-bert',
    'config': {
        'epochs': 3,
        # 'bert_checkpoint': "Babelscape/wikineural-multilingual-ner",
        'bert_checkpoint': "Yaxin/xlm-roberta-base-conll2003-ner",
        # 'bert_checkpoint': "tner/xlm-roberta-base-conll2003",
        # 'bert_checkpoint': "FacebookAI/xlm-roberta-large",
        # 'bert_checkpoint': "google-bert/bert-base-cased",
        'learning_rate': 5e-5,
        'max_grad_norm': 0.99,
        'weight_decay': 0.1,
        'per_device_train_batch_size': 16,
        'use_bilstm': True,
        'use_crf': True,
        'bert_layers': 12,
        'dropout_rate': 0.1
    }
}

def main():
    """
    Main function to execute model training and evaluation.
    """
    with wandb.init(**wandb_args):
        config = wandb.config
        # Set random seeds for reproducibility
        set_random_seeds()

        # Initialize tokenizer and model
        model = initialize_model(config)

        # Initialize tokenizer
        tokenizer = initialize_tokenizer(config)

        raw_datasets = build_dataset()

        assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"

        # Load and preprocess datasets
        tokenized_datasets = preprocess_data(raw_datasets, tokenizer)

        # Build and train the model
        trainer = build_trainer(
            model=model,
            tokenizer=tokenizer,
            tokenized_datasets=tokenized_datasets,
            config=config,
        )
        trainer.train()

        # Evaluate the model on the test dataset
        test_metrics = trainer.evaluate(
            eval_dataset=tokenized_datasets["test"],
            metric_key_prefix="test",
        )
        wandb.log(test_metrics)

        print("Test Metrics:", test_metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
