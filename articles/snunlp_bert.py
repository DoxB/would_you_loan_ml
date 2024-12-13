import pandas as pd
import numpy as np
import datetime as dt
# from konlpy.tag import Okt
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import VotingClassifier
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np

##데이터 불러오기
df = pd.read_excel('NewsResult_20201205-20241205.xlsx')
df_copy = df.copy()

##SNU감정분석
# 1. TPU 설정
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU 확인
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU initialized.")
except ValueError:
    strategy = tf.distribute.get_strategy()  # TPU가 없으면 기본 전략 사용
    print("TPU not available, using default strategy.")

# 2. 모델과 토크나이저 로드 (TPU 내에서 로드)
MODEL_NAME = "snunlp/KR-FinBert-SC"
with strategy.scope():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, from_pt=True)

# 3. 감정 분석 함수 (중립 임계값 조정 및 키워드 강화 포함)
def detailed_sentiment_analysis(text, neutral_threshold=0.7, positive_keywords=None, negative_keywords=None):
    # 키워드 기본값 설정
    if positive_keywords is None:
        positive_keywords = ["호황", "급등", "상승", "활황", "회복"]
    if negative_keywords is None:
        negative_keywords = ["침체", "폭락", "급감", "하락", "위기"]

    # 사전 키워드 분석
    text_lower = text.lower()
    for keyword in positive_keywords:
        if keyword in text_lower:
            return "positive"
    for keyword in negative_keywords:
        if keyword in text_lower:
            return "negative"

    # 모델 기반 감정 분석
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    scores = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]

    # 감정 점수 추출
    negative_score = scores[0]
    neutral_score = scores[1]
    positive_score = scores[2]

    # 임계값 기반 감정 결정
    if neutral_score > neutral_threshold:
        return "neutral"
    elif positive_score > negative_score:
        return "positive"
    else:
        return "negative"

# 4. 배치 처리 함수 (TPU 효율성을 위한 배치 처리 추가)
def batch_sentiment_analysis(texts, batch_size=32, neutral_threshold=0.7, positive_keywords=None, negative_keywords=None):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = []
        for text in batch:
            label = detailed_sentiment_analysis(
                text, neutral_threshold=neutral_threshold,
                positive_keywords=positive_keywords, negative_keywords=negative_keywords
            )
            batch_results.append(label)
        results.extend(batch_results)
    return results

# 5. 데이터프레임에 함수 적용 (배치 처리)
texts = df_copy['제목'].tolist()
df_copy['label'] = batch_sentiment_analysis(
    texts, batch_size=32, neutral_threshold=0.6,
    positive_keywords=["상승", "활발", "회복"],
    negative_keywords=["폭락", "하락", "침체"]
)

# 6. 라벨을 숫자로 변환
df_copy['label'].replace({'positive': 1, 'neutral': 0, 'negative': -1}, inplace=True)

# 7. 라벨 분포 확인
print(df_copy['label'].value_counts())

#8. 저장
df_copy.to_csv('final_label_result.csv')


##전처리
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF 벡터화
print("\nTF-IDF 벡터화 중...")
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


##SVM
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF 벡터화
print("\nTF-IDF 벡터화 중...")
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


##RandomForest
from sklearn.ensemble import RandomForestClassifier

# 모델 학습
print("RandomClassifier 모델 학습 중...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
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
plt.close()


##Catboost
from catboost import CatBoostClassifier

# 모델 학습
print("Catboost 모델 학습 중...")
model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=42, class_weights=[5.0, 1.0, 5.0])
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

# # 혼동 행렬 시각화
# plt.figure(figsize=(8, 6))
# sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.close()


##앙상블
# 앙상블 모델 정의
# 로지스틱 회귀, SVC, XGBoost, CatBoost 모델 사용
svc_clf = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state = 42)  # 확률 출력 활성화
cat_clf = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=42, class_weights=[5.0, 1.0, 5.0])
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# VotingClassifier를 사용해 앙상블 구성
ensemble_model = VotingClassifier(
    estimators=[('svc', svc_clf), ('catboost', cat_clf), ('randomforest', rf_clf)],
    voting='soft'  # soft voting 사용 (확률 기반)
)

# 모델 학습
ensemble_model.fit(X_train_tfidf, y_train)

# 예측
y_pred = ensemble_model.predict(X_test_tfidf)

# 결과 평가
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
plt.show()
