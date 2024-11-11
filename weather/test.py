import requests
import json

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

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    # JSON 데이터를 파일로 저장
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("JSON 데이터가 data.json 파일에 저장되었습니다.")
else:
    print("Failed to retrieve data:", response.status_code)
