from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Dict

from constants import MODEL_CHECKPOINT


def initialize_tokenizer(config: Dict) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    Initialize a tokenizer for token classification.

    Returns:
        A tokenizer for token classification.
    NOTE: Below is an example of how to initialize a tokenizer for token classification. You are free to change this.
    # But make sure the tokenizer is the same as the model.
    """
    model_name = config.get("bert_checkpoint", MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
