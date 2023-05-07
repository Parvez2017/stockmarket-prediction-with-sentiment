import numpy as np
import pandas as pd 
from transformers import BertTokenizer
from datasets import Dataset

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, classification_report

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def prepare_dataset(df):
    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=512), batched=True)
    dataset.set_format(type='numpy', columns=['input_ids', 'label'])
    X = dataset["input_ids"]
    y = dataset["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    classification_scores = classification_report(labels, predictions, output_dict=True)
    return {'accuracy' : accuracy_score(predictions, labels),
            'classification_scores': classification_scores}


if __name__ == "__main__":
    df_path = '/home/parvej/work/stockmarket-prediction-with-sentiment/final_data.tsv'
    df = pd.read_csv(df_path, sep="\t") ## use your own customized dataset
    df = df[["body", "change"]]
    df = df.rename(columns={"body": "sentence", "change": "label"})
    df = df.dropna(subset=['sentence', 'label']) ## drop missing values
   
    X_train, X_test, y_train, y_test = prepare_dataset(df)

    MNB = MultinomialNB()
    MNB.fit(X_train, y_train)

    predicted = MNB.predict(X_test)
    accuracy_score = accuracy_score(predicted, y_test)

    print('MultinominalNB model accuracy is',str('{:04.2f}'.format(accuracy_score*100))+'%')
    print('------------------------------------------------')
    print(classification_report(y_test, predicted))