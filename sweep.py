import os

from dataset import build_dataset, preprocess_data
from model import initialize_model
from tokenizer import initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds

from test import test_dataset

import wandb

# os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'eval/f1',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        's': 2,
        'eta': 3
    }
}

parameters_dict = {
    'epochs': {
        'values': [10]
    },
    'bert_checkpoint': {
        'values': [
            "google-bert/bert-base-cased",
            "dslim/bert-base-NER",
            "dslim/bert-large-NER",
            "dslim/distilbert-NER",
            "Babelscape/wikineural-multilingual-ner"
        ]
    },
    'learning_rate': {
        'distribution': 'uniform',
        'min': 5e-6,
        'max': 5e-5
    },
    'max_grad_norm': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 0.99
    },
    'weight_decay': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
    },
    'per_device_train_batch_size': {
        'values': [8, 16, 32]
    },
    'use_bilstm': {
        'values': [True, False]
    },
    'use_crf': {
        'values': [True, False]
    },
    'bert_layers': {
        'values': [2, 4, 6, 8, 10, 12]
    },
    'dropout_rate': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.5
    }
}

sweep_config['parameters'] = parameters_dict

def sweep_train(config=None):
    """
    Sweep train function for hyperparameter tuning.
    """
    with wandb.init(config=None):
        config = wandb.config
        # Set random seeds for reproducibility
        set_random_seeds()

        # Initialize tokenizer and model
        model = initialize_model(config['use_bilstm'], config['use_crf'], config['dropout_rate'])

        # Initialize tokenizer
        tokenizer = initialize_tokenizer()

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

def main():
    """
    Main function to execute model training and evaluation.
    """
    # Build and train the model
    sweep_id = wandb.sweep(sweep_config, project="ner")
    wandb.agent(sweep_id, function=sweep_train, count=50)
    # # Evaluate the model on the test dataset
    # test_metrics = trainer.evaluate(
    #     eval_dataset=tokenized_datasets["test"],
    #     metric_key_prefix="test",
    # )

    # print("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()
