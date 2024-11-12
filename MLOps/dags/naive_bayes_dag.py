from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

def train_naive_bayes():
    data = pd.read_csv('/path/to/your/data.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    model = GaussianNB()
    model.fit(X, y)

    joblib.dump(model, '/path/to/save/naive_bayes_model.pkl')

def evaluate_naive_bayes():
    model = joblib.load('/path/to/save/naive_bayes_model.pkl')
    data = pd.read_csv('/path/to/your/data.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f'Naive Bayes Accuracy: {accuracy}')

with DAG('naive_bayes_dag', start_date=datetime(2023, 1, 1), schedule_interval='@daily', catchup=False) as dag:
    train_task = PythonOperator(task_id='train_naive_bayes', python_callable=train_naive_bayes)
    evaluate_task = PythonOperator(task_id='evaluate_naive_bayes', python_callable=evaluate_naive_bayes)

    train_task >> evaluate_task
