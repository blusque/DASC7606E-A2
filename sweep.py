import os

from dataset import build_dataset, preprocess_data
from model import initialize_model
from tokenizer import initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds

import wandb

# os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'test_f1',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        's': 1,
        'eta': 3
    }
}

parameters_dict = {
    'epochs': {
        'values': [ 3, 6, 9, 12 ]
    },
    'bert_checkpoint': {
        'values': [ "Tirendaz/multilingual-xlm-roberta-for-ner" ]
    },
    'learning_rate': {
        'values': [ 2e-5 ]
    },
    'lr_scheduler_type': {
        'values': [ 'linear' ]
    },
    'max_grad_norm': {
        'values': [ 1.0 ]
    },
    'warmup_ratio': {
        'values': [ 0.1 ],
    },
    'weight_decay': {
        'values': [ 0.05 ]
    },
    'per_device_train_batch_size': {
        'values': [ 16 ],
    },
    'use_bilstm': {
        'values': [ True ]
    },
    'use_crf': {
        'values': [ True ]
    },
    'bert_layers': {
        'values': [ 10, 12 ]
    },
    'dropout_rate': {
        'values': [ 0.1 ]
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

        print("Test Metrics:", test_metrics)
        wandb.log(test_metrics)
        wandb.finish()

def main():
    """
    Main function to execute model training and evaluation.
    """
    # Build and train the model
    sweep_id = wandb.sweep(sweep_config, project="ner")
    wandb.agent(sweep_id, function=sweep_train, count=12)
    # # Evaluate the model on the test dataset
    # test_metrics = trainer.evaluate(
    #     eval_dataset=tokenized_datasets["test"],
    #     metric_key_prefix="test",
    # )

    # print("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()
