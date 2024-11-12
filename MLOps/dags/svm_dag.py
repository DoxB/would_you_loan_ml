from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

def train_svm():
    data = pd.read_csv('/path/to/your/data.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    model = SVC()
    model.fit(X, y)

    joblib.dump(model, '/path/to/save/svm_model.pkl')

def evaluate_svm():
    model = joblib.load('/path/to/save/svm_model.pkl')
    data = pd.read_csv('/path/to/your/data.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f'SVM Accuracy: {accuracy}')

with DAG('svm_dag', start_date=datetime(2023, 1, 1), schedule_interval='@daily', catchup=False) as dag:
    train_task = PythonOperator(task_id='train_svm', python_callable=train_svm)
    evaluate_task = PythonOperator(task_id='evaluate_svm', python_callable=evaluate_svm)

    train_task >> evaluate_task
