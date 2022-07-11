import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from code_function import preprocess_script, make_dataset, reduction_dataset
from sklearn.model_selection import train_test_split , KFold, StratifiedKFold

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-unimodal")

import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained("D:/code_classification/uni_aug/checkpoint-28162")
model.to(device)

tokenizer.truncation_side = 'left'

MAX_LEN = 1024
INPUT = {"train": "D:/code_classification/python3_train16.csv", "test": "D:/code_classification/python3_valid16.csv"}

from datasets import load_dataset, load_metric
dataset = load_dataset("csv", data_files=INPUT)

def example_fn(examples):
    outputs = tokenizer(examples['code1'], examples['code2'], padding=True, max_length=MAX_LEN, truncation=True)
    if 'similar' in examples:
        outputs["labels"] = examples["similar"]
    return outputs


dataset = dataset.map(example_fn, remove_columns=['code1', 'code2', 'similar'])

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
_collator = DataCollatorWithPadding(tokenizer=tokenizer)
_metric = load_metric("glue", "sst2")
args = TrainingArguments(
    'D:/code_classification/uni_aug',
    # load_best_model_at_end = True,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    do_train=True,
    do_eval=True,
    fp16 = True,
    optim="adafactor",
    save_strategy="epoch",
    logging_strategy="epoch",
    evaluation_strategy="epoch")

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def metric_fn2(p):
    preds, labels = p
    output =  _metric.compute(references=labels, predictions=np.where(sigmoid(preds)>0.5, 1, 0))
    return output
class MyTrainer(Trainer):
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name # 각인!
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_name == 'BinaryEntropy':
            custom_loss = torch.nn.BCEWithLogitsLoss()

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            loss = custom_loss(outputs[0], labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
trainer = MyTrainer(
        loss_name='BinaryEntropy',
        model=model,
        args=args,
        data_collator=_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=metric_fn2,
        )

TEST =  "D:/code_classification/indent_test.csv"
test_dataset = load_dataset("csv", data_files=TEST)['train']
test_dataset = test_dataset.map(example_fn, remove_columns=['code1', 'code2'])

test_prediction = trainer.predict(test_dataset)
test_pred = sigmoid(test_prediction.predictions)

np.save('D:/code_classification/uni_aug/prediction2.npy', arr=test_pred)
result = np.where(test_pred >= 0.5, 1, 0)
sample_submission = pd.read_csv("D:/code_classification/sample_submission.csv")
sample_submission['similar'] = result
sample_submission.to_csv('D:/code_classification/sub0610_2.csv', index=False)