from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import json
import re
import os

# 기본 경로 설정
BASE_PATH = '/opt/airflow/dags'
DATA_PATH = os.path.join(BASE_PATH, 'data', 'weighted_labeled_news.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'models', 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_PATH, 'models', 'tfidf_vectorizer.pkl')
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

def preprocess_text(text):
    """텍스트 전처리 함수"""
    if pd.isna(text):
        return ''
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

def check_data_quality(**context):
    """데이터 품질 체크"""
    print("데이터 품질 검사 시작...")
    
    # 데이터 로드
    df = pd.read_csv(DATA_PATH)
    
    # 필수 컬럼 확인
    required_columns = ['제목', '키워드', '특성추출(가중치순 상위 50개)', '감성라벨']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")
    
    # 결측치 확인
    null_ratios = df[required_columns].isnull().mean()
    if (null_ratios > 0.1).any():
        raise ValueError(f"높은 결측치 비율 발견: {null_ratios[null_ratios > 0.1]}")
    
    # 레이블 값 확인
    valid_labels = [0, 1, 2]
    invalid_labels = df[~df['감성라벨'].isin(valid_labels)]['감성라벨'].unique()
    if len(invalid_labels) > 0:
        raise ValueError(f"잘못된 레이블 값 발견: {invalid_labels}")
    
    print("데이터 품질 검사 완료")
    return "data_quality_passed"

def train_evaluate_model(**context):
    """모델 학습 및 평가"""
    print("모델 학습 시작...")
    
    # 데이터 로드 및 전처리
    df = pd.read_csv(DATA_PATH)
    df['combined_text'] = df[['제목', '키워드', '특성추출(가중치순 상위 50개)']].fillna('').agg(' '.join, axis=1)
    df['combined_text'] = df['combined_text'].apply(preprocess_text)
    
    # 특성 추출
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(df['combined_text'])
    y = df['감성라벨']
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE 오버샘플링
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    # 모델 학습
    model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
    model.fit(X_train_resampled, y_train_resampled)
    
    # 예측 및 평가
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    # 결과 출력
    print(f"\nAccuracy: {accuracy}")
    print("\nClassification Report:\n", report)
    
    # 모델과 벡터라이저 저장
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)
    
    # 메트릭스 저장
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics = {
        'accuracy': float(accuracy),
        'report': report,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_size': len(df)
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)
    
    print("모델 학습 및 평가 완료")
    return metrics

def check_model_performance(**context):
    """모델 성능 체크"""
    print("모델 성능 확인 시작...")
    
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        
        current_accuracy = metrics['accuracy']
        print(f"현재 모델 정확도: {current_accuracy}")
        
        # 성능이 임계값 이하면 재학습 필요
        if current_accuracy < 0.75:
            print("모델 성능이 임계값 미달, 재학습 필요")
            return "train_model"  # "retrain_model"이 아니라 실제 태스크 ID인 "train_model"을 반환
        else:
            print("모델 성능 양호")
            return "skip_training"
        
    except FileNotFoundError:
        print("기존 모델 메트릭스 없음, 최초 학습 진행")
        return "train_model"  # 초기 학습을 위해 "train_model" 반환


def predict_new_data(**context):
    """새로운 데이터에 대한 예측"""
    print("새로운 데이터 예측 시작...")
    
    # 모델과 벡터라이저 로드
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # 새로운 데이터 로드 (예시)
    # new_df = pd.read_csv('path_to_new_data.csv')
    # 여기에서 실제 새로운 데이터에 대한 예측 로직 구현
    
    print("새로운 데이터 예측 완료")

# DAG 정의
with DAG('sentiment_analysis_pipeline',
         default_args=default_args,
         description='뉴스 감성분석 파이프라인',
         schedule_interval='@daily',
         catchup=False) as dag:
    
    start = EmptyOperator(task_id='start')
    
    check_data = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality
    )
    
    check_performance = BranchPythonOperator(
        task_id='check_model_performance',
        python_callable=check_model_performance
    )
    
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_evaluate_model
    )
    
    skip_train = EmptyOperator(task_id='skip_training')
    
    predict = PythonOperator(
        task_id='predict_new_data',
        python_callable=predict_new_data
    )
    
    end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed'
    )
    
    # 태스크 순서 설정
    start >> check_data >> check_performance
    check_performance >> [train_model, skip_train] >> predict >> end