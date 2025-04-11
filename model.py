from constants import MODEL_CHECKPOINT, ID_TO_LABEL, LABEL_TO_ID
import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from typing import List, Optional, Dict
import re
from CRF import CRF

class BertForNER(nn.Module):
    def __init__(self, bert_model_checkpoint, num_labels, dropout=0.5, use_bilstm=False, use_crf=False, **kwargs):
        super(BertForNER, self).__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained(
            bert_model_checkpoint,
            num_labels=num_labels,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
            output_hidden_states=True,
            return_dict=True,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        layer_len = self.bert.config.num_hidden_layers
        assert layer_len >= 12, 'bert model should have at least 12 layers'

        self.use_bilstm = use_bilstm
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.trained_layers = kwargs.get('bert_layers', 12)

        
        if self.use_bilstm:
            lstm_output_size = kwargs.pop('lstm_output_size', 256)
            self.bilstm = nn.LSTM(self.bert.config.hidden_size, lstm_output_size // 2, bidirectional=True, batch_first=True)
            self.output = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(lstm_output_size, num_labels)
            )
        else:
            self.output = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.bert.config.hidden_size, num_labels)
            )

        if self.use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        
        self.loss = nn.CrossEntropyLoss()

        self.is_eval = False
        self.init_weights()
        
        trained_layer_list = []
        for i in range(layer_len - 1, layer_len - self.trained_layers - 1, -1):
            trained_layer_list.append('encoder.layer.' + str(i))
        for name, param in self.bert.named_parameters():
            result = re.search('encoder.layer.1*[0-9]', name)
            if result is not None and result.group() in trained_layer_list:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def init_weights(self):
        if self.use_bilstm:
            self.bilstm.reset_parameters()
        torch.nn.init.xavier_uniform_(self.output[1].weight)
        if self.use_crf:
            self.crf.reset_parameters()

    def eval(self):
        super().eval()
        self.is_eval = True

    def train(self, mode=True):
        super().train(mode)
        self.is_eval = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels
        )
        # return outputs
        sequence_output = outputs.hidden_states[-1]
        # print(labels)
        if self.use_bilstm:
            sequence_output, _ = self.bilstm(sequence_output)
            logits = self.output(sequence_output)
        else:
            logits = self.output(sequence_output)

        if self.use_crf:
            mask = (labels[:, 1:] != -100).to(logits.device)
            # logits[:, 1:] = self.crf.decode(logits[:, 1:], mask=mask)
            loss = -self.crf(logits[:, 1:], labels[:, 1:], mask=mask, reduction='mean')
        else:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        
        if self.is_eval:
            if self.use_crf:
                mask = (labels[:, 1:] != -100).to(logits.device)
                crf_result = self.crf.decode(logits[:, 1:], mask=mask)
                logits[:, 1:] = torch.zeros_like(logits[:, 1:]).to(logits.device)
                for i in range(len(crf_result)):
                    for j in range(len(crf_result[i])):
                        logits[i, j + 1, crf_result[i][j]] = 1
                # logits[torch.arange(logits.size(0)).unsqueeze(1), torch.arange(1, logits.size(1) - 1).unsqueeze(0), crf_result] = 1

        return TokenClassifierOutput(loss=loss, logits=logits)

def initialize_model(config: Dict) -> nn.Module:
    """
    Initialize a model for token classification.

    Returns:
        A model for token classification.

    NOTE: Below is an example of how to initialize a model for token classification.

    from transformers import AutoModelForTokenClassification
    from constants import ID_TO_LABEL, LABEL_TO_ID, MODEL_CHECKPOINT

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    You are free to change this.
    But make sure the model meets the requirements of the `transformers.Trainer` API.
    ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer
    """
    # Write your code here.

    # model = AutoModelForTokenClassification.from_pretrained(
    #     pretrained_model_name_or_path=MODEL_CHECKPOINT,
    #     num_labels=len(LABEL_TO_ID),  # Ensure the number of labels matches LABEL_TO_ID
    #     id2label=ID_TO_LABEL,
    #     label2id=LABEL_TO_ID,
    #     ignore_mismatched_sizes=True,  # Ignore size mismatches
    # )
    model_name = config.get('bert_checkpoint', MODEL_CHECKPOINT)
    use_bilstm = config.get('use_bilstm', False)
    use_crf = config.get('use_crf', False)
    dropout_prob = config.get('dropout_rate', 0.1)
    bert_layers = config.get('bert_layers', 12)

    model = BertForNER(model_name, len(LABEL_TO_ID), use_bilstm=use_bilstm, use_crf=use_crf, dropout=dropout_prob, bert_layers=bert_layers)

    return model