from random import sample
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding


@dataclass
class DataModule:

    cfg: dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError

        self.raw_dataset = load_dataset(
            self.cfg["dataset_name"], self.cfg["dataset_config"]
        )
        self.labels = self.raw_dataset["train"].features["labels"].feature.names
        self.num_labels = len(self.labels)
        self.label2int = {label: i for i, label in enumerate(self.labels)}
        self.int2label = {v: k for k, v in self.label2int.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_name_or_path"],
        )

        self.label_tokens = [f"[{label}]" for label in self.labels]

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.label_tokens}
        )

    def prepare_dataset(self):

        if self.cfg["load_from_disk"]:
            self.tokenized_dataset = load_from_disk(self.cfg["load_from_disk"])
        else:
            num_train_examples = len(self.raw_dataset["train"])

            for i in sample(list(range(num_train_examples)), k=5):
                x = self.raw_dataset["train"][i]
                print(x["text"])
                print([self.int2label[i] for i in x["labels"]])

            def tokenize(examples):
                texts = ["".join(self.label_tokens + [t]) for t in examples["text"]]
                tokenized = self.tokenizer(
                    texts, 
                    padding=False, 
                    truncation=True,
                    max_length=self.cfg["max_seq_length"],
                )

                num_examples = len(examples["labels"])
                onehot_labels = np.zeros((num_examples, self.num_labels), dtype=np.float32)

                for row_num, labels in enumerate(examples["labels"]):
                    onehot_labels[row_num, labels] = 1

                tokenized["labels"] = onehot_labels

                return tokenized

            self.tokenized_dataset = self.raw_dataset.map(
                tokenize, batched=True, num_proc=self.cfg["num_proc"]
            )
            
            
            


@dataclass
class DataCollatorFloatLabels(DataCollatorWithPadding):
    def __call__(self, *args, **kwargs):
        batch = super().__call__(*args, **kwargs)

        batch["labels"] = batch.pop("labels").float()

        return batch