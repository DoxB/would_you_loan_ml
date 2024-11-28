from flask import Flask, jsonify
import mysql.connector

app = Flask(__name__)

# 데이터베이스 설정
DATABASES = {
    "shinhan_loan": {
        "host": "regularmark.iptime.org",
        "port": 436,  # shinhan 컨테이너의 포트
        "user": "",
        "password": "",
        "database": "shinhan_loan"
    },
    "kookmin_loan": {
        "host": "regularmark.iptime.org",
        "port": 434,  # Kookmin 컨테이너의 포트
        "user": "",
        "password": "",
        "database": "kookmin_loan"
    },
    "etc_loans": {
        "host": "regularmark.iptime.org",
        "port": 437,  # ETC 컨테이너의 포트
        "user": "",
        "password": "",
        "database": "etc_loan"  # 올바른 데이터베이스 이름
    },
    "woori_loans": {
        "host": "regularmark.iptime.org",
        "port": 435,  # Woori 컨테이너의 포트
        "user": "",
        "password": "",
        "database": "woori_loan"  # 올바른 데이터베이스 이름
    }
}

# 각 데이터베이스에서 조회할 테이블 설정
VALID_TABLES = {
    "shinhan_loan": ["shinhan_loans"],  # Shinhan 스키마
    "kookmin_loan": ["kookmin_loans"],
    "etc_loans": ["etc_loans"],  # ETC 테이블 이름
    "woori_loans": ["woori_loans"]  # Woori 테이블 이름
}

# Helper 함수: 데이터베이스 연결 및 쿼리 실행
def query_db(db_config, query):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    except mysql.connector.Error as e:
        return {"error": str(e)}

# API 엔드포인트: 모든 데이터 조회
@app.route('/loans', methods=['GET'])
def get_all_loans():
    """
    모든 데이터베이스와 모든 테이블의 데이터를 가져옵니다.
    """
    all_results = {}
    for db_name, db_config in DATABASES.items():
        # 데이터베이스에 정의된 모든 테이블 조회
        table_names = VALID_TABLES.get(db_name, [])
        for table_name in table_names:
            query = f"SELECT * FROM {table_name};"
            results = query_db(db_config, query)
            if "error" not in results:
                all_results[f"{db_name}.{table_name}"] = results
            else:
                all_results[f"{db_name}.{table_name}"] = {"error": results["error"]}
    return jsonify(all_results), 200

# 메인 실행
if __name__ == '__main__':
    app.run(debug=True, port=5011)
