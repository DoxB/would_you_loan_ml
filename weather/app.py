from flask import Flask, render_template, jsonify
import requests
import json

app = Flask(__name__)

def get_housing_data():
    url = "https://data-api.kbland.kr/bfmstat/wthrchat/husePrcIndx"
    params = {
        "월간주간구분코드": "01",
        "매매전세코드": "01",
        "매물종별구분": "01",
        "면적크기코드": "00",
        "단위구분코드": "01",
        "법정동코드": "",
        "지역명": "전국",
        "시도명": "전국",
        "조회시작일자": "202410",
        "조회종료일자": "202410",
        "selectedTab": "0",
        "changeRatio": "true",
        "mapType": "false",
        "페이지번호": "",
        "페이지목록수": "",
        "zoomLevel": "8",
        "탭구분코드": "0"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/housing-data')
def housing_data():
    data = get_housing_data()
    if data and 'dataBody' in data and 'data' in data['dataBody'] and 'depth2' in data['dataBody']['data']:
        processed_data = []
        for item in data['dataBody']['data']['depth2']:
            # 서울(11)과 경기도(41)의 데이터만 필터링
            if (str(item['법정동코드']).startswith(('11', '41')) and 
                item['데이터여부'] == 1):
                
                weather = '☀️'  # 기본값
                if float(item['변동률']) < 0:
                    weather = '☁️'  # 하락시 흐림
                
                # 지역명 앞에 [서울] 또는 [경기] 표시 추가
                region_prefix = '[서울] ' if str(item['법정동코드']).startswith('11') else '[경기] '
                
                processed_data.append({
                    'name': region_prefix + item['지역명'],
                    'lat': float(item['wgs84중심위도']),
                    'lng': float(item['wgs84중심경도']),
                    'change': float(item['변동률']),
                    'weather': weather,
                    'current_value': float(item['현재데이터']),
                    'prev_value': float(item['전기데이터'])
                })
        return jsonify(processed_data)
    return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)