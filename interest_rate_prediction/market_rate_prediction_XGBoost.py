from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

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
    'xgboost_hyperparameter_tuning',
    default_args=default_args,
    description='XGBoost Hyperparameter Tuning with Airflow',
    schedule_interval='@daily',
    catchup=False,
)

# 하이퍼파라미터 목록 정의
hyperparameters = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.01, "subsample": 0.9},
    {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05, "subsample": 0.7},
]

# 데이터 로드 함수
def load_data(**context):
    data = pd.read_csv('/opt/airflow/dags/data/features.csv')
    context['task_instance'].xcom_push(key='raw_data', value=data.to_dict())
    return "Data loaded successfully"

# 데이터 전처리 함수
def prepare_data(**context):
    ti = context['task_instance']
    data = pd.DataFrame.from_dict(ti.xcom_pull(task_ids='load_data', key='raw_data'))
    
    df = data.set_index('지표').T
    df = df.apply(lambda x: pd.to_numeric(x.str.replace(',', ''), errors='coerce'))
    
    target = df['시장금리_CD'].values  # 예: '시장금리_CD'를 예측
    features = df.drop(columns=['시장금리_CD']).values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 학습 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42, shuffle=False)
    
    # 데이터 저장
    context['task_instance'].xcom_push(key='prepared_data', value={
        'X_train': X_train.tolist(),
        'X_test': X_test.tolist(),
        'y_train': y_train.tolist(),
        'y_test': y_test.tolist()
    })
    return "Data prepared successfully"

def train_model(hyperparameters, **context):
    """XGBoost 모델 학습"""
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    ti = context['task_instance']
    data = pd.DataFrame.from_dict(ti.xcom_pull(task_ids='prepare_data', key='processed_data'))
    
    # 특성 및 타겟 변수 정의
    X = data[feature_columns]
    y = data['시장금리_CD']
    
    # 데이터의 결측값 및 이상값 확인 및 제거
    X = X.fillna(0)  # 결측값을 0으로 대체하거나, mean()으로 대체 가능
    y = y.replace([np.inf, -np.inf], np.nan).dropna()  # 무한값을 NaN으로 변환 후 제거
    X = X.loc[y.index]  # 타겟 값이 제거된 경우 해당 행도 X에서 제거
    
    # 학습 및 테스트 데이터 분리
    test_size = 12
    train_size = len(X) - test_size
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    # 데이터 스케일링 (선택 사항)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost 모델 정의
    model = xgb.XGBRegressor(
        n_estimators=hyperparameters['n_estimators'],
        max_depth=hyperparameters['max_depth'],
        learning_rate=hyperparameters['learning_rate'],
        colsample_bytree=hyperparameters['colsample_bytree'],
        subsample=hyperparameters['subsample'],
        random_state=42
    )
    
    # 모델 학습
    model.fit(
        X_train_scaled, 
        y_train, 
        eval_set=[(X_test_scaled, y_test)], 
        early_stopping_rounds=10, 
        verbose=False
    )
    
    # 평가
    predictions = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    # 결과 저장
    result = {
        'hyperparameters': hyperparameters,
        'rmse': rmse,
        'mae': mae
    }
    logger.info(f"Model trained with RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # 결과를 XCom에 저장
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
