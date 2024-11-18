from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# DAG 기본 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'random_forest_hyperparameter_tuning',
    default_args=default_args,
    description='Random Forest Hyperparameter Tuning with Airflow',
    schedule_interval='@daily',
    catchup=False,
)

# 하이퍼파라미터 조합 정의
hyperparameters = [
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 4},
]

# 데이터 로드 함수
def load_data(**context):
    data = pd.read_csv('/opt/airflow/dags/data/features.csv')
    context['task_instance'].xcom_push(key='raw_data', value=data.to_dict())
    return "Data loaded successfully"

def prepare_data(**context):
    """데이터 전처리"""
    ti = context['task_instance']
    data = pd.DataFrame.from_dict(ti.xcom_pull(task_ids='load_data', key='raw_data'))
    
    # 지표를 인덱스로 설정하고 전치
    df = data.set_index('지표').T
    
    # 데이터를 numeric으로 변환
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    
    # lag features 추가
    for feature in feature_columns:
        df[f'{feature}_lag1'] = df[feature].shift(1)
        df[f'{feature}_lag3'] = df[feature].shift(3)
    
    # NA 제거 또는 대체
    imputer = SimpleImputer(strategy='mean')  # 결측값을 평균값으로 대체
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # 데이터 준비 완료
    context['task_instance'].xcom_push(key='processed_data', value=df.to_dict())
    return "Data prepared successfully"
 
def train_model(hyperparameters, **context):
    """랜덤 포레스트 모델 학습"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # XCom에서 데이터 가져오기
    ti = context['task_instance']
    data = pd.DataFrame.from_dict(ti.xcom_pull(task_ids='prepare_data', key='processed_data'))
    
    # 특성 및 타겟 변수 정의
    X = data[feature_columns]
    y = data['시장금리_CD']
    
    # 학습 및 테스트 데이터 분리
    test_size = 12
    train_size = len(data) - test_size
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 정의 및 하이퍼파라미터 설정
    rf = RandomForestRegressor(
        n_estimators=hyperparameters['n_estimators'],
        max_depth=hyperparameters['max_depth'],
        min_samples_split=hyperparameters['min_samples_split'],
        min_samples_leaf=hyperparameters['min_samples_leaf'],
        random_state=42
    )
    
    # 모델 학습
    rf.fit(X_train_scaled, y_train)
    
    # 평가
    predictions = rf.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    # 결과 저장
    result = {
        'hyperparameters': hyperparameters,
        'rmse': rmse,
        'mae': mae
    }
    context['task_instance'].xcom_push(key='model_result', value=result)
    
    return f"Model trained with {hyperparameters}. RMSE: {rmse:.4f}, MAE: {mae:.4f}"


# 결과 로깅 함수
def log_results(**context):
    ti = context['task_instance']
    all_results = []
    
    for i in range(len(hyperparameters)):
        result = ti.xcom_pull(task_ids=f'train_model_{i}', key='model_result')
        all_results.append(result)
    
    # 결과를 로그로 출력
    for res in all_results:
        print(f"Hyperparameters: {res['hyperparameters']}, RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}")
    
    return "Results logged successfully"

# Task 정의
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    dag=dag,
)

train_tasks = []
for i, params in enumerate(hyperparameters):
    train_task = PythonOperator(
        task_id=f'train_model_{i}',
        python_callable=train_model,
        op_kwargs={'hyperparameters': params},
        dag=dag,
    )
    train_tasks.append(train_task)

log_results_task = PythonOperator(
    task_id='log_results',
    python_callable=log_results,
    dag=dag,
)

# Task 의존성 정의
load_data_task >> prepare_data_task
for train_task in train_tasks:
    prepare_data_task >> train_task
train_tasks >> log_results_task
