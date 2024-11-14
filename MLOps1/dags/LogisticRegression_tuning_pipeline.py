from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import json
import itertools
import os

# 경로 설정
BASE_PATH = '/opt/airflow/dags'
DATA_PATH = os.path.join(BASE_PATH, 'data', 'weighted_labeled_news.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'models')
METRICS_PATH = os.path.join(BASE_PATH, 'metrics', 'model_metrics.json')

# DAG 기본 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 하이퍼파라미터 그리드
param_grid = {
    'C': [0.1, 1, 10],
    'max_iter': [100, 200]
}

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    df['combined_text'] = df[['제목', '키워드', '특성추출(가중치순 상위 50개)']].fillna('').agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(df['combined_text'])
    y = df['감성라벨']
    joblib.dump(tfidf, os.path.join(MODEL_PATH, 'tfidf_vectorizer.pkl'))
    return X, y

def train_model_with_params(params, **context):
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    model = LogisticRegression(C=params['C'], max_iter=params['max_iter'], random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    model_path = os.path.join(MODEL_PATH, f"model_C_{params['C']}_max_iter_{params['max_iter']}.pkl")
    joblib.dump(model, model_path)
    
    metrics = {
        'params': params,
        'accuracy': accuracy
    }
    
    # 메트릭스 파일에 JSON 객체를 줄바꿈으로 구분하여 추가
    with open(METRICS_PATH, 'a') as f:
        f.write(json.dumps(metrics) + "\n")
    
    print(f"모델 학습 완료: {params} -> 정확도: {accuracy}")

def choose_best_params(**context):
    best_accuracy = 0
    best_params = None
    
    with open(METRICS_PATH, 'r') as f:
        for line in f:
            metrics = json.loads(line)
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_params = metrics['params']
    
    print(f"최적의 하이퍼파라미터: {best_params} -> 정확도: {best_accuracy}")
    return best_params

# DAG 정의
with DAG(
    'tuning_pipeline_LR',
    default_args=default_args,
    description='하이퍼파라미터 튜닝 파이프라인',
    schedule_interval='*/60 * * * *',  # 60분마다 실행
    catchup=False
) as dag:
    
    start = EmptyOperator(task_id='start')
    
    # 모든 하이퍼파라미터 조합에 대해 train_model_with_params 태스크 생성
    for params in itertools.product(param_grid['C'], param_grid['max_iter']):
        param_dict = {'C': params[0], 'max_iter': params[1]}
        train_task = PythonOperator(
            task_id=f'train_model_C_{params[0]}_max_iter_{params[1]}',
            python_callable=train_model_with_params,
            op_kwargs={'params': param_dict}
        )
        start >> train_task
    
    # 최적의 하이퍼파라미터 선택 태스크
    choose_best = PythonOperator(
        task_id='choose_best_params',
        python_callable=choose_best_params
    )

    start >> choose_best
