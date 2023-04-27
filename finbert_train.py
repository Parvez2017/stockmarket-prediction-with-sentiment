import numpy as np
import pandas as pd 
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-pretrain',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')


def prepare_dataset(df_train, df_val, df_test):
    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)
    dataset_test = Dataset.from_pandas(df_test)

    dataset_train = dataset_train.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=512), batched=True)
    dataset_val = dataset_val.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=512), batched=True)
    dataset_test = dataset_test.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length' , max_length=512), batched=True)

    dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    dataset_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    return dataset_train, dataset_val, dataset_test

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy' : accuracy_score(predictions, labels)}


if __name__ == "__main__":
    df_path = '/home/parvej/work/stockmarket-prediction-with-sentiment/final_data.tsv'
    df = pd.read_csv(df_path, sep="\t") ## use your own customized dataset
    df = df[["body", "change"]]
    df = df.rename(columns={"body": "sentence", "change": "label"})

    df = df.dropna(subset=['sentence', 'label']) ## drop missing values

    df_train, df_test, = train_test_split(df, stratify=df['label'], test_size=0.1, random_state=42)
    df_train, df_val = train_test_split(df_train, stratify=df_train['label'],test_size=0.1, random_state=42)
    print(df_train.shape, df_test.shape, df_val.shape)

    args = TrainingArguments(
            output_dir = 'temp/',
            evaluation_strategy = 'steps',
            save_strategy = 'steps',
            save_steps=10000,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=500,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
    )
    dataset_train, dataset_val, dataset_test = prepare_dataset(df_train, df_val, df_test)

    trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=args,                  # training arguments, defined above
            train_dataset=dataset_train,         # training dataset
            eval_dataset=dataset_val,            # evaluation dataset
            compute_metrics=compute_metrics
    )

    trainer.train()   