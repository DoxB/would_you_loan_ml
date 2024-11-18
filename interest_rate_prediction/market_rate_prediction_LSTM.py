from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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
    'lstm_hyperparameter_tuning',
    default_args=default_args,
    description='LSTM Hyperparameter Tuning with Airflow',
    schedule_interval='@daily',
    catchup=False,
)

# 하이퍼파라미터 목록 정의
hyperparameters = [
    {"units": 50, "dropout": 0.2, "batch_size": 32, "learning_rate": 0.001},
    {"units": 100, "dropout": 0.3, "batch_size": 64, "learning_rate": 0.0001},
    {"units": 150, "dropout": 0.4, "batch_size": 16, "learning_rate": 0.01},
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
    
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 시계열 데이터 생성
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i-sequence_length:i])
        y.append(target[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # 학습 및 테스트 데이터 분할
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 데이터 저장
    context['task_instance'].xcom_push(key='prepared_data', value={
        'X_train': X_train.tolist(),
        'X_test': X_test.tolist(),
        'y_train': y_train.tolist(),
        'y_test': y_test.tolist()
    })
    return "Data prepared successfully"

def train_model(hyperparameters, **context):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # XCom에서 데이터 가져오기
    ti = context['task_instance']
    data = ti.xcom_pull(task_ids='prepare_data', key='prepared_data')
    
    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    X_test = np.array(data['X_test'])
    y_test = np.array(data['y_test'])
    
    # 하이퍼파라미터 설정
    units = hyperparameters['units']
    dropout = hyperparameters['dropout']
    learning_rate = hyperparameters['learning_rate']
    
    # 모델 정의
    model = Sequential([
        LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=dropout),
        Dense(1)
    ])
    
    # Optimizer에 learning_rate 설정
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')  # 수정된 부분
    
    # 모델 학습
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    # 평가
    predictions = model.predict(X_test)
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
        print(f"Hyperparameters: {res['hyperparameters']}, Loss: {res['loss']:.4f}")
    
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
