# 시장금리(CD,CP) feature들 상관관계 분석 // 11.18.(월) 오전/오후

<img width="490" alt="image" src="https://github.com/user-attachments/assets/0245e1cc-793f-4741-bd9b-031e549ba233">

<img width="1023" alt="image" src="https://github.com/user-attachments/assets/6927b369-4a6c-4055-9bff-126d7dce65a8">

<img width="1034" alt="image" src="https://github.com/user-attachments/assets/5d60e275-37a0-493f-af88-74a9e21add7b">

<img width="731" alt="image" src="https://github.com/user-attachments/assets/5af43488-970d-47a8-bf02-ae23f4a9db9e">
CD/CP 금리와의 상관관계 (금리 변수 제외)]
지표               시장금리_CD   시장금리_CP
지표                                
시장금리_CD         1.000000  0.987156
시장금리_CP         0.987156  1.000000
수출물가지수          0.565426  0.584185
비경제활동인구_30-39세  0.540669  0.499732
비경제활동인구_15-29세  0.537892  0.501514
비경제활동인구_15-19세  0.504545  0.459895
비경제활동인구_20-29세  0.398247  0.401304
경제심리지수_순환변동치    0.319516  0.204538
경제심리지수_원계열      0.289776  0.165949
건설기성액_국내외국기관   -0.130872 -0.160576
건설기성액_공공기관     -0.333635 -0.320267
수입물가지수         -0.338711 -0.304657
비경제활동인구_40-49세 -0.369107 -0.351826
전산업생산지수_건설업    -0.416612 -0.386113
건설기성액_민자       -0.474467 -0.437983
국내건설수주액_단위_백만원 -0.500399 -0.450209
전산업생산지수_공공행정   -0.549824 -0.501704
건설기성액_민간기관     -0.579776 -0.529031
건설기성액_총액       -0.604719 -0.555299
전산업생산지수_서비스업   -0.682317 -0.640522
경기종합지수_후행종합지수  -0.686462 -0.641285
경기종합지수_선행종합지수  -0.696757 -0.652792
소매판매액지수        -0.704793 -0.666243
비경제활동인구_60세이상  -0.705034 -0.656319
소비자물가지수        -0.707131 -0.666890
전산업생산지수_총계     -0.707749 -0.670831
경기종합지수_동행종합지수  -0.730948 -0.690598
전산업생산지수_광공업    -0.741630 -0.717541
비경제활동인구_총계     -0.761688 -0.716915
비경제활동인구_50-59세 -0.770124 -0.744443

<img width="787" alt="image" src="https://github.com/user-attachments/assets/80baeadb-e4db-411a-ab5e-93d6dae297fa">

<img width="789" alt="image" src="https://github.com/user-attachments/assets/0502a10a-e005-47b8-88fc-59a8b01ab132">

=== CD 금리 예측 성능 ===

학습 세트
RMSE: 0.1229
MAE: 0.0808
R2: 0.9916

테스트 세트
RMSE: 0.1108
MAE: 0.1005
R2: 0.1107

=== CP 금리 예측 성능 ===

학습 세트
RMSE: 0.1493
MAE: 0.0872
R2: 0.9890

테스트 세트
RMSE: 0.2923
MAE: 0.2065
R2: -0.5619

<img width="788" alt="image" src="https://github.com/user-attachments/assets/70ef1512-a652-4f23-b795-85dd9987b6a8">

<img width="791" alt="image" src="https://github.com/user-attachments/assets/6f50c2a5-c605-4d1a-bb0e-f0ed14d0321c">


---

# Sentiment Analysis Model Training Pipeline // 11.14.(목) 오전

이 프로젝트는 뉴스 감성 분석을 위해 다양한 머신러닝 모델을 학습하고, 하이퍼파라미터 튜닝을 수행하는 Airflow 파이프라인을 포함하고 있습니다. 주요 알고리즘은 **로지스틱 회귀(Logistic Regression)**, **서포트 벡터 머신(SVM)**, 그리고 **랜덤 포레스트(Random Forest)**입니다. 각 알고리즘은 Airflow 파이프라인으로 실행되며, 모델의 성능을 평가하고 최적의 하이퍼파라미터를 찾는 방식으로 동작합니다.

<img width="1226" alt="image" src="https://github.com/user-attachments/assets/4283b3dd-7149-47f6-9187-1b6a1bade362">

---

## 1. Logistic Regression Pipeline (LogisticRegression_tuning_pipeline.py)
### 설명
로지스틱 회귀를 사용하여 뉴스 감성 분석 모델을 학습합니다. `C` (정규화 강도)와 `max_iter` (최대 반복 횟수) 하이퍼파라미터를 그리드 탐색으로 최적화합니다. 모델의 성능은 정확도로 평가하며, 최적의 하이퍼파라미터 조합을 선택하여 저장합니다.

### 주요 구성 요소
- **하이퍼파라미터 그리드**: `C`, `max_iter`
- **모델 평가 지표**: 정확도(Accuracy)
- **Airflow 스케줄링**: 60분마다 실행되도록 설정됨 (`*/60 * * * *`)

### 파일 구조
- 저장 경로: `/models/logistic_regression`


## 2. Support Vector Machine Pipeline (svm_tuning_pipeline.py)
### 설명
서포트 벡터 머신(SVM)을 사용하여 뉴스 감성 분석 모델을 학습합니다. `C` (정규화 강도)와 `kernel` (커널 함수) 하이퍼파라미터를 조정하여 모델의 최적 성능을 찾습니다. 정확도에 따라 각 모델을 평가하며, 가장 높은 정확도를 기록한 모델을 선택합니다.

### 주요 구성 요소
- **하이퍼파라미터 그리드**: `C`, `kernel`
- **모델 평가 지표**: 정확도(Accuracy)
- **Airflow 스케줄링**: 60분마다 실행되도록 설정됨 (`*/60 * * * *`)

### 파일 구조
- 저장 경로: `/models/svm`

## 3. Random Forest Pipeline (RandomForestClassifier_tuning_pipeline.py)
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
