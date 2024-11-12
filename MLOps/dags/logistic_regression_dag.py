from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

def train_logistic_regression():
    # 데이터 로드
    data = pd.read_csv('/path/to/your/data.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    # 모델 학습
    model = LogisticRegression()
    model.fit(X, y)

    # 모델 저장
    joblib.dump(model, '/path/to/save/logistic_regression_model.pkl')

def evaluate_logistic_regression():
    # 데이터 로드 및 모델 평가
    model = joblib.load('/path/to/save/logistic_regression_model.pkl')
    data = pd.read_csv('/path/to/your/data.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f'Logistic Regression Accuracy: {accuracy}')

with DAG('logistic_regression_dag', start_date=datetime(2023, 1, 1), schedule_interval='@daily', catchup=False) as dag:
    train_task = PythonOperator(task_id='train_logistic_regression', python_callable=train_logistic_regression)
    evaluate_task = PythonOperator(task_id='evaluate_logistic_regression', python_callable=evaluate_logistic_regression)

    train_task >> evaluate_task
