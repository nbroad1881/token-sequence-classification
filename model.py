from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from transformers import (
    PreTrainedModel,
    AutoModel,
)
from transformers.modeling_outputs import ModelOutput


class AutoModelForTokenSequenceClassification(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.transformer = AutoModel.from_config(config, **kwargs)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 1)

    @classmethod
    def from_pretrained(cls, model_name_or_path, config, use_auth_token=None):
        """
        If model_name_or_path points to a local file, it will load from state dict.
        Otherwise, it will pull the transformer weights and randomly initialize
        the classifier weights.
        """

        model_name_or_path = Path(model_name_or_path)

        if (
            str(model_name_or_path).endswith("pytorch_model.bin")
            or (model_name_or_path / "pytorch_model.bin").exists()
        ):
            model_name_or_path = (
                model_name_or_path
                if model_name_or_path.name == "pytorch_model.bin"
                else model_name_or_path / "pytorch_model.bin"
            )
            model = cls(config)
            model.load_state_dict(torch.load(model_name_or_path))
            return model

        model = cls(config)
        model.transformer = AutoModel.from_pretrained(
            model_name_or_path, config=config, use_auth_token=use_auth_token
        )

        model.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if model.classifier.bias is not None:
            model.classifier.bias.data.zero_()

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        token_type_ids=None,
        **kwargs
    ):
        
        sequence_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        sequence_output = self.dropout(sequence_output[0])

        logits = self.classifier(sequence_output)[
            :, 1 : 1 + self.config.num_labels, :
        ].squeeze()

        loss = None
        if labels is not None:

            if self.config.problem_type == "multi_label_classification":
                loss = nn.BCEWithLogitsLoss()(logits, labels)
            else:
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, self.config.num_labels), labels.view(-1)
                )
        if self.config.problem_type == "multi_label_classification":
            probas = logits.sigmoid()    
        else:
            probas = logits.softmax(dim=-1)

        if self.config.return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "probas": probas,
            }
        
        return SequenceClassificationOutput(
            loss=loss,
            logits=logits,
            probas=probas,
        )

    def set_embeddings(self, method="avg", tokenizer=None):
        """
        if method == "avg" this sets the new embeddings to the average of all of the old
        embeddings.
        if method == "similar", then the tokenizer needs to be passed. The average value of the 
        label embeddings will be used for the new tokens. If a label is 'neutral', then the new 
        embedding will be the same as `new_value` as shown below
        label_ids = tokenizer("neutral", add_special_tokens=False).input_ids
        new_value = self.model.transformer.word_embeddings.weight[label_ids].mean()

        TODO: set to value of CLS, average with CLS
        """

        num_old_embeddings = self.model.transformer.word_embeddings.weight.data.size(0) - self.config.num_labels

        if method == "avg":
            avg_value = self.model.transformer.embeddings.word_embeddings.weight.data[:num_old_embeddings].mean(dim=0)

            self.model.transformer.word_embeddings.weight.data[num_old_embeddings:] = avg_value
        
        elif method == "similar":

            for label in self.config.label2int.keys():
                # literal_ids are the ids for the tokens if the label were put through the tokenizer
                # tokenizer("neutral")

                literal_ids = tokenizer(label.lower(), add_special_tokens=False, return_tensors="pt").input_ids
                avg_value = self.model.transformer.word_embeddings.weight.data[literal_ids].mean(0)

                # new_id is the new token id that was added
                new_id = tokenizer(f"[{label}]", add_special_tokens=False).input_ids
                self.model.transformer.word_embeddings.weight[new_id] = avg_value


@dataclass
class SequenceClassificationOutput(ModelOutput):

    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    probas: torch.FloatTensor = None
    
