import sys
import warnings
import logging
import argparse
from functools import partial

import transformers
from transformers import Trainer, TrainingArguments, AutoConfig
import datasets
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback
from transformers.trainer_utils import get_last_checkpoint

from data import DataModule, DataCollatorFloatLabels
from model import AutoModelForTokenSequenceClassification
from utils import compute_metrics, set_wandb_env_vars, get_configs, NewWandbCB

# change logging to not be bombarded by messages
# if you are debugging, the messages will likely be helpful
# warnings.simplefilter("ignore")
# logging.disable(logging.WARNING)

logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("-l", "--load_from_disk", type=str, default=None)

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()
    

    cfg, train_args = get_configs(args.config_file)
    set_wandb_env_vars(cfg)

    training_args = TrainingArguments(**train_args)
    cfg["load_from_disk"] = args.load_from_disk
    
    

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_module = DataModule(cfg)

    data_module.prepare_dataset()

    dataset = data_module.tokenized_dataset

    config = AutoConfig.from_pretrained(
        cfg["model_name_or_path"],
        num_labels=len(data_module.int2label),
        problem_type=cfg["problem_type"],
    )

    model = AutoModelForTokenSequenceClassification.from_pretrained(
        cfg["model_name_or_path"],
        config=config,
    )
    
    model.transformer.resize_token_embeddings(len(data_module.tokenizer))

    if cfg["problem_type"] == "multi_label_classification":
        data_collator = DataCollatorFloatLabels(data_module.tokenizer, pad_to_multiple_of=cfg["pad_multiple"])
    else:
        data_collator = DataCollatorWithPadding(data_module.tokenizer, pad_to_multiple_of=cfg["pad_multiple"])

    comp_met = partial(compute_metrics, problem_type=cfg["problem_type"])

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=comp_met,
        tokenizer=data_module.tokenizer,
        data_collator=data_collator,
        callbacks=[NewWandbCB(cfg)],
    )

    trainer.remove_callback(WandbCallback)

    # Training
    if training_args.do_train:

        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset["train"])

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if training_args.do_predict:
        logger.info("*** Predict ***")

        metrics = trainer.predict(dataset["test"]).metrics
        metrics["eval_samples"] = len(dataset["test"])

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    kwargs = {
        "finetuned_from": cfg["model_name_or_path"],
        "tasks": "text-classification",
        "language": "en",
        "dataset_tags": cfg["dataset_name"],
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()