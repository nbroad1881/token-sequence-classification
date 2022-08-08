import warnings
import logging

from transformers import Trainer, TrainingArguments, AutoConfig

from model import AutoModelForTokenSequenceClassification
from data import DataModule

# change logging to not be bombarded by messages
# if you are debugging, the messages will likely be helpful
warnings.simplefilter("ignore")
logging.disable(logging.WARNING)


