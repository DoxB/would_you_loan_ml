# Sentiment Analysis Model Training Pipeline

이 프로젝트는 뉴스 감성 분석을 위해 다양한 머신러닝 모델을 학습하고, 하이퍼파라미터 튜닝을 수행하는 Airflow 파이프라인을 포함하고 있습니다. 주요 알고리즘은 **로지스틱 회귀(Logistic Regression)**, **서포트 벡터 머신(SVM)**, 그리고 **랜덤 포레스트(Random Forest)**입니다. 각 알고리즘은 Airflow 파이프라인으로 실행되며, 모델의 성능을 평가하고 최적의 하이퍼파라미터를 찾는 방식으로 동작합니다.

<img width="1226" alt="image" src="https://github.com/user-attachments/assets/4283b3dd-7149-47f6-9187-1b6a1bade362">

---

## 1. Logistic Regression Pipeline (logistic_regression_pipeline.py)
### 설명
로지스틱 회귀를 사용하여 뉴스 감성 분석 모델을 학습합니다. `C` (정규화 강도)와 `max_iter` (최대 반복 횟수) 하이퍼파라미터를 그리드 탐색으로 최적화합니다. 모델의 성능은 정확도로 평가하며, 최적의 하이퍼파라미터 조합을 선택하여 저장합니다.

### 주요 구성 요소
- **하이퍼파라미터 그리드**: `C`, `max_iter`
- **모델 평가 지표**: 정확도(Accuracy)
- **Airflow 스케줄링**: 60분마다 실행되도록 설정됨 (`*/60 * * * *`)

### 파일 구조
- 저장 경로: `/models/logistic_regression`


## 2. Support Vector Machine Pipeline (svm_pipeline.py)
### 설명
서포트 벡터 머신(SVM)을 사용하여 뉴스 감성 분석 모델을 학습합니다. `C` (정규화 강도)와 `kernel` (커널 함수) 하이퍼파라미터를 조정하여 모델의 최적 성능을 찾습니다. 정확도에 따라 각 모델을 평가하며, 가장 높은 정확도를 기록한 모델을 선택합니다.

### 주요 구성 요소
- **하이퍼파라미터 그리드**: `C`, `kernel`
- **모델 평가 지표**: 정확도(Accuracy)
- **Airflow 스케줄링**: 60분마다 실행되도록 설정됨 (`*/60 * * * *`)

### 파일 구조
- 저장 경로: `/models/svm`

## 3. Random Forest Pipeline (random_forest_pipeline.py)
### 설명
랜덤 포레스트를 사용하여 뉴스 감성 분석 모델을 학습합니다. 하이퍼파라미터 그리드에는 `n_estimators` (트리 개수), `max_depth` (트리 최대 깊이), `min_samples_split` (노드를 분할하기 위한 최소 샘플 수)가 포함됩니다. 정확도를 기준으로 각 모델을 평가하며, 가장 높은 정확도를 가진 모델을 최종 선택합니다.

### 주요 구성 요소
- **하이퍼파라미터 그리드**: `n_estimators`, `max_depth`, `min_samples_split`
- **모델 평가 지표**: 정확도(Accuracy)
- **Airflow 스케줄링**: 60분마다 실행되도록 설정됨 (`*/60 * * * *`)

### 파일 구조
- 저장 경로: `/models/random_forest`

---

### 공통 사항

각 파이프라인은 SMOTE를 사용하여 학습 데이터의 불균형을 조정하며, 모델 학습에 필요한 TF-IDF 벡터라이저를 생성 및 저장합니다. 각 모델은 평가 후 최적의 하이퍼파라미터와 정확도를 JSON 형식으로 저장합니다.


----

# MLOps 구축 // 11.12.(화) 오후
1. 뉴스기사 라벨링
2. 모델 : 1) 로지스틱 회귀 2) SVM 3) Navive 모델 사용
3. 진행중---

<img width="1090" alt="image" src="https://github.com/user-attachments/assets/8367b8c3-3d28-40bc-b6c9-a5f3c4e10320">




# 구별 뉴스기사 감정분석 (부동산 리포트 전용 기사) // 11.12.(화) 오전

<img width="585" alt="image" src="https://github.com/user-attachments/assets/bd837ca4-6ff4-4ad8-8f42-2063255a8f37">

<img width="485" alt="image" src="https://github.com/user-attachments/assets/e6684054-c285-4aae-82e3-f3e56b09a85e">


<img width="751" alt="image" src="https://github.com/user-attachments/assets/3e84ef74-f0fa-456f-8808-ddfdc7bbcbfe">

---

# 부동산 날씨 생성완료[경기~서울] / 11.11.(월)

- 지도 줌에 따라 나타낼 데이터를 보여줘야함
  
<img width="1010" alt="image" src="https://github.com/user-attachments/assets/11aa9e43-e1cb-41af-bbe3-15fddd21c946">

---
# 소득예측 모델 폐기...
# 1. 소비 카테고리별 Cluster 군집화 진행

## 1-1. 다음과 같은 카테고리별로 군집화를 진행함 
카테고리를 나는 기준 참고 문헌: [(보도자료)2023년+2분기+가계동향조사+결과.pdf](https://github.com/user-attachments/files/17671185/2023.%2B2.%2B.%2B.pdf)

- 성별
- 나이
- 카드 총 사용액
- 신용카드 총 사용액
- 체크카드 총 사용액
- 식료품·비주류음료
- 의류·신발
- 주거·수도·광열
- 보건
- 교통
- 통신
- 오락·문화
- 교육
- 음식·숙박
- 기타

## 1-2. k-means 엘보 스코어 결과

- k값 3 ~ 10 군집화 진행

![image](https://github.com/user-attachments/assets/ff9542e4-f133-451c-9927-f33866df32e9)

- 최적의 k값은 5개

## 1-3. 결과

<img width="580" alt="image" src="https://github.com/user-attachments/assets/adc381cd-fd94-4200-a75c-3529269f8e36">

각 클러스터 별 SEQ 갯수

- Cluster 0: 143531 SEQ
- Cluster 1: 364771 SEQ
- Cluster 2: 160898 SEQ
- Cluster 3: 35941 SEQ
- Cluster 4: 90215 SEQ


# 2. 카드 등급을 사용한 소득 분위 예측

## 2-1. 클러스터별 카드 등급 갯수

<img width="378" alt="image" src="https://github.com/user-attachments/assets/baaa7ea2-6f43-4423-b449-a67459bce801">

- 21: VVIP
- 22: VIP
- 23: 플래티넘
- 24: 골드
- 25: 해당없음 


## 2-2. 카드 등급결로 가중치를 부여
- 21: VVIP -> 5점 
- 22: VIP -> 4점
- 23: 플래티넘 -> 3점
- 24: 골드 -> 2점
- 25: 해당없음 -> 1점


## 2-3. 가중치 부여 이후 Cluster별로 갯수 평균을 구함

<img width="378" alt="image" src="https://github.com/user-attachments/assets/a221a707-d196-4f04-bdfa-997f9a9a600d">


## 2-4. 결과
- 평군값이 높은 값이 나올수록 높은 분위로 할당

- Cluster 3 : 5분위
- Cluster 0 : 4분위
- Cluster 4 : 3분위
- Cluster 2 : 2분위
- Cluster 1 : 1분위
