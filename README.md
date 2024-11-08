# 1. 카테고리별 Cluster 군집화
다음과 같은 카테고리별로 군집화를 진행함 
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
- 기타*

<img width="580" alt="image" src="https://github.com/user-attachments/assets/adc381cd-fd94-4200-a75c-3529269f8e36">

# 2. Cluster 별 카드 등급 Count
카드 데이터의 카드 등급을 이용하여 클러스터 진행
- 21: VVIP
- 22: VIP
- 23: 플래티넘
- 24: 골드
- 25: 해당없음 

<img width="378" alt="image" src="https://github.com/user-attachments/assets/baaa7ea2-6f43-4423-b449-a67459bce801">

# 3. 카드 등급으로 Cluster 결정

## 3-1. 카드 등급결로 가중치를 부여
- 21: VVIP -> 5점 
- 22: VIP -> 4점
- 23: 플래티넘 -> 3점
- 24: 골드 -> 2점
- 25: 해당없음 -> 1점
## 3-2. 각 Cluster의 평균을 구함
<img width="378" alt="image" src="https://github.com/user-attachments/assets/a221a707-d196-4f04-bdfa-997f9a9a600d">
## 3-3. 결과
- 평군값이 높은 값이 나올수록 높은 분위로 할당 
<img width="226" alt="image" src="https://github.com/user-attachments/assets/85e6b22f-940a-4308-bbfb-462901b07b96">
- Cluster 3 : 5분위
- Cluster 0 : 4분위
- Cluster 4 : 3분위
- Cluster 2 : 2분위
- Cluster 1 : 1분위
