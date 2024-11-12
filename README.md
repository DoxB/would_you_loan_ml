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
