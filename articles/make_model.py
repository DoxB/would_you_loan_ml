import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

def load_data(file_path='final_label_result.csv'):
    """데이터 로드 함수"""
    df = pd.read_csv(file_path)
    print(f"데이터 로드 완료: {len(df)} 행")
    return df

def preprocess_data(df):
    """데이터 전처리 함수"""
    # # NaN 열 처리
    # df = df.drop(columns = ['사건/사고 분류1', '사건/사고 분류2', '사건/사고 분류3'])

    df = df.sort_values(by = '일자', ascending=True)

    # NaN 값 처리
    df['제목'] = df['제목'].fillna('')
    df['키워드'] = df['키워드'].fillna('')
    df['특성추출(가중치순 상위 50개)'] = df['특성추출(가중치순 상위 50개)'].fillna('')
    
    # 문자열 타입으로 변환
    df['제목'] = df['제목'].astype(str)
    df['키워드'] = df['키워드'].astype(str)
    df['통합 분류3'] = df['통합 분류3'].astype(str)
    df['특성추출(가중치순 상위 50개)'] = df['특성추출(가중치순 상위 50개)'].astype(str)
    
    # 특성 결합
    df['combined_features'] = df['제목'] + ' ' + df['키워드'] + ' ' + df['특성추출(가중치순 상위 50개)']
    
    # 빈 특성 제거
    df = df[df['combined_features'].str.strip() != '']
    
    X = df['combined_features']
    y = df['label']
    
    print("\n전처리 결과:")
    print(f"전처리 후 데이터 개수: {len(df)}")
    print(f"감성 라벨 분포:\n{y.value_counts()}")
    
    return X, y

def create_save_directory():
    """저장 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'saved_models_{timestamp}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def train_and_save_model(X, y, save_dir):
    """모델 학습 및 저장"""
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF 벡터화
    print("\nTF-IDF 벡터화 중...")
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 모델 학습
    print("앙상블 모델 학습 중...")
    # 로지스틱 회귀, SVC, XGBoost, CatBoost 모델 사용
    svc_clf = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state = 42)  # 확률 출력 활성화
    cat_clf = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=42, class_weights=[10.0, 1.0, 10.0])
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',scale_pos_weight=5, random_state=42)

    # VotingClassifier를 사용해 앙상블 구성
    model = VotingClassifier(
        estimators=[('svc', svc_clf), ('catboost', cat_clf), ('xgb', xgb_clf)],
        voting='soft'  # soft voting 사용 (확률 기반)
    )
    model.fit(X_train_tfidf, y_train)
    
    # 예측 및 평가
    print("\n모델 평가 중...")
    y_pred = model.predict(X_test_tfidf)
    
    # 평가 지표 계산
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_precision': precision_score(y_test, y_pred, average='macro'),
        'macro_recall': recall_score(y_test, y_pred, average='macro'),
        'macro_f1': f1_score(y_test, y_pred, average='macro')
    }
    
    # 평가 결과 출력
    print("\n=== 모델 성능 평가 ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()
    
    # 모델 저장
    print(f"\n모델 저장 중... 저장 위치: {save_dir}")
    joblib.dump(model, f'{save_dir}/sentiment_model.pkl')
    joblib.dump(tfidf, f'{save_dir}/tfidf_vectorizer.pkl')
    
    # 평가 지표 저장
    with open(f'{save_dir}/metrics.txt', 'w', encoding='utf-8') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    return model, tfidf, metrics

def test_model(model, tfidf):
    """모델 테스트"""
    test_texts = [
        {
            "제목": "긍정적인 경제 성장률 전망",
            "키워드": "경제,성장,전망",
            "특성": "경제 성장 긍정 전망 상승"
        },
        {
            "제목": "환경오염 심각성 증가",
            "키워드": "환경,오염,문제",
            "특성": "환경 오염 문제 심각 우려"
        }
    ]
    
    print("\n테스트 예측 결과:")
    for text in test_texts:
        combined = f"{text['제목']} {text['키워드']} {text['특성']}"
        prediction = model.predict(tfidf.transform([combined]))[0]
        print(f"\n입력: {text['제목']}")
        print(f"예측 감성: {prediction}")

def main():
    # 1. 데이터 로드
    print("데이터 로드 중...")
    df = load_data()
    
    # 2. 데이터 전처리
    print("\n데이터 전처리 중...")
    X, y = preprocess_data(df)
    
    # 3. 저장 디렉토리 생성
    save_dir = create_save_directory()
    
    # 4. 모델 학습 및 저장
    model, tfidf, metrics = train_and_save_model(X, y, save_dir)
    
    # 5. 모델 테스트
    test_model(model, tfidf)
    
    print(f"\n작업 완료! 모델과 관련 파일들이 {save_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()