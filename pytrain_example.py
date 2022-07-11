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
model = RobertaForSequenceClassification.from_pretrained("microsoft/unixcoder-base-unimodal", num_labels=1)
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

trainer.train()
# TEST =  "D:/code_classification/indent_test.csv"
# test_dataset = load_dataset("csv", data_files=TEST)['train']
# test_dataset = test_dataset.map(example_fn, remove_columns=['code1', 'code2'])
#
# test_prediction = trainer.predict(test_dataset)
# test_pred = sigmoid(test_prediction.predictions)
# np.save('D:/code_classification/uni_aug/prediction.npy', arr=test_pred)



'''
{'eval_loss': 0.03271318972110748, 'eval_accuracy': 0.9489034843593455, 'eval_runtime': 88.7645, 'eval_samples_per_second': 101.2, 'eval_steps_per_second': 12.651, 'epoch': 1.0}


{'eval_loss': 0.022343425080180168, 'eval_accuracy': 0.9740621173327396, 'eval_runtime': 88.4518, 'eval_samples_per_second': 101.558, 'eval_steps_per_second': 12.696, 'epoch': 2.0}


{'loss': 0.006, 'learning_rate': 2.001207300617854e-05, 'epoch': 3.0}


{'eval_loss': 0.021036116406321526, 'eval_accuracy': 0.9772904374930425, 'eval_runtime': 88.7339, 'eval_samples_per_second': 101.235, 'eval_steps_per_second': 12.656, 'epoch': 3.0}

{'loss': 0.0035, 'learning_rate': 1.0016334067182729e-05, 'epoch': 4.0}


{'eval_loss': 0.018828047439455986, 'eval_accuracy': 0.9809640431926974, 'eval_runtime': 88.5139, 'eval_samples_per_second': 101.487, 'eval_steps_per_second': 12.687, 'epoch': 4.0}

{'loss': 0.0022, 'learning_rate': 1.988495135288687e-08, 'epoch': 5.0}

{'eval_loss': 0.017597228288650513, 'eval_accuracy': 0.983413113659134, 'eval_runtime': 88.4376, 'eval_samples_per_second': 101.574, 'eval_steps_per_second': 12.698, 'epoch': 5.0}

{'train_runtime': 39319.392, 'train_samples_per_second': 28.649, 'train_steps_per_second': 1.791, 'train_loss': 0.01182364217344904, 'epoch': 5.0}
'''