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


    def model_init():
        config = AutoConfig.from_pretrained(
            cfg["model_name_or_path"],
            num_labels=len(data_module.int2label),
            problem_type=cfg["problem_type"],
            return_dict=True,
        )
        model =  AutoModelForTokenSequenceClassification.from_pretrained(
            cfg["model_name_or_path"],
            config=config,
        )
    
    
        model.transformer.resize_token_embeddings(len(data_module.tokenizer))
        
        return model

    if cfg["problem_type"] == "multi_label_classification":
        data_collator = DataCollatorFloatLabels(data_module.tokenizer, pad_to_multiple_of=cfg["pad_multiple"])
    else:
        data_collator = DataCollatorWithPadding(data_module.tokenizer, pad_to_multiple_of=cfg["pad_multiple"])

    comp_met = partial(compute_metrics, problem_type=cfg["problem_type"])

    # Initialize our Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=comp_met,
        tokenizer=data_module.tokenizer,
        data_collator=data_collator,
        callbacks=[NewWandbCB(cfg)],
    )

    trainer.remove_callback(WandbCallback)
    
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 7e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 24, 32]),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.009, 0.02]),
            "adam_epsilon": trial.suggest_categorical("adam_epsilon", [1e-6, 1e-7, 1e-8]),
        }

    def optuna_objective(metrics):
        return metrics["eval_f1"]

    # using same trainer as above
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        n_trials=25,
        hp_space=optuna_hp_space,
        compute_objective=optuna_objective
    )


if __name__ == "__main__":
    main()