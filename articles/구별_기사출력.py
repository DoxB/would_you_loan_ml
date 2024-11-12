import streamlit as st
import pandas as pd
from collections import defaultdict

def load_and_process_data(file_path):
    # 서울 구 리스트
    seoul_districts = [
        '강남구', '강동구', '강북구', '강서구', '관악구', '광진구',
        '구로구', '금천구', '노원구', '도봉구', '동대문구', '동작구',
        '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구',
        '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구'
    ]
    
    # 경기도 구 리스트
    gyeonggi_districts = [
        '수정구', '중원구', '분당구',  # 성남시
        '장안구', '권선구', '팔달구', '영통구',  # 수원시
        '만안구', '동안구',  # 안양시
        '단원구', '상록구',  # 안산시
        '덕양구', '일산동구', '일산서구',  # 고양시
        '처인구', '기흥구', '수지구',  # 용인시
        '원미구', '소사구', '부천구'  # 부천시
    ]
    
    # CSV 파일 읽기
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # 결과를 저장할 딕셔너리
    district_news = {
        'seoul': defaultdict(list),
        'gyeonggi': defaultdict(list)
    }
    
    # 각 행을 순회하며 분석
    for idx, row in df.iterrows():
        locations = str(row['위치']).split(',') if pd.notna(row['위치']) else []
        news_info = {
            'title': row['제목'],
            'date': row['일자']
        }
        
        # 서울 구 체크
        for district in seoul_districts:
            if any(district in loc for loc in locations):
                district_news['seoul'][district].append(news_info)
        
        # 경기도 구 체크
        for district in gyeonggi_districts:
            if any(district in loc for loc in locations):
                district_news['gyeonggi'][district].append(news_info)
    
    return district_news

def main():
    st.title('서울/경기 구별 부동산 뉴스')
    
    # 데이터 로드
    file_path = "NewsResult_20231112-20241112.csv"
    district_news = load_and_process_data(file_path)
    
    # 지역 선택
    region = st.radio("지역 선택", ["서울", "경기도"])
    region_key = 'seoul' if region == "서울" else 'gyeonggi'
    
    # 구 선택
    districts = sorted(district_news[region_key].keys())
    
    # 각 구별 뉴스 건수를 포함하여 표시
    district_counts = {d: len(district_news[region_key][d]) for d in districts}
    districts_with_counts = [f"{d} ({district_counts[d]}건)" for d in districts]
    
    selected_district_with_count = st.selectbox(
        f'{region} 구 선택',
        districts_with_counts
    )
    
    # 선택된 구 이름 추출 (건수 제외)
    selected_district = selected_district_with_count.split(' (')[0]
    
    # 선택된 구의 뉴스 표시
    if selected_district:
        st.markdown(f"### 최근 기사 Top 5")
        news_list = district_news[region_key][selected_district]
        
        # 최신순으로 정렬하여 상위 5개만 표시
        news_list = sorted(news_list, key=lambda x: x['date'], reverse=True)[:5]
        
        # 뉴스 목록을 깔끔하게 표시
        for i, news in enumerate(news_list, 1):
            st.write(f"{i}. [{news['date']}] {news['title']}")
        
        # 전체 기사 수 표시
        total_news = len(district_news[region_key][selected_district])
        if total_news > 5:
            st.write(f"\n... 외 {total_news - 5}건의 기사가 더 있습니다.")

if __name__ == "__main__":
    main()
