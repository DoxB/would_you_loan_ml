# 실행방법
# uvicorn app.main:app --reload --port 30049 --host 0.0.0.0

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import pickle
import numpy as np
import os
from pathlib import Path

app = FastAPI()

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(BASE_DIR, 'credit_model.pkl')

# DB 연결 설정
username = "root"
password = "0000"
engine = create_engine(f"mysql+pymysql://{username}:{password}@regularmark.iptime.org:43306/web_test")
SessionLocal = sessionmaker(bind=engine)

@app.get("/credit-scores")
async def get_credit_scores():
    try:
        with SessionLocal() as session:
            # DB에서 데이터 가져오기
            df = pd.read_sql_query("SELECT * FROM web_test.credit_score_evaluation", session.connection())
            
            # 컬럼명 매핑
            column_mapping = {
                'tot_use_am': 'TOT_USE_AM',
                'ues_income': 'UES_INCOME',
                'crdsl_use_am': 'CRDSL_USE_AM',
                'cnf_use_am': 'CNF_USE_AM',
                'plsanit_am': 'PLSANIT_AM',
                'fsbz_am': 'FSBZ_AM',
                'trvlec_am': 'TRVLEC_AM',
                'dan_rt': 'DAN_RT',
                'dist_am': 'DIST_AM',
                'life_stage_dan': 'LIFE_STAGE_DAN',
                'clothgds_am': 'CLOTHGDS_AM',
                'att_ym': 'ATT_YM',
                'debt': 'DEBT',
                'debt_ratio': 'DEBT_RATIO',
                'income': 'INCOME'
            }
            
            # 필요한 컬럼만 선택하고 이름 변경
            model_df = df[list(column_mapping.keys())].rename(columns=column_mapping)
            model_df['quarter_seq'] = df['quarter_seq']

            print(f"Loading model from {MODEL_PATH}")  # 디버깅용 출력
            # 신용평가 모델 실행
            with open(MODEL_PATH, 'rb') as f:
                model_info = pickle.load(f)

            coefficients = model_info['coefficients']
            scaler = model_info['scaler']
            required_features = model_info['required_features']

            X = model_df[required_features]
            X_scaled = scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=required_features, index=model_df.index)
            X_scaled.insert(0, 'const', 1)

            logit_score = pd.Series(np.dot(X_scaled, coefficients), index=model_df.index)
            percentile_ranks = logit_score.rank(pct=True)

            results = []
            for idx in model_df.index:
                rank = percentile_ranks[idx]
                if rank >= 0.5506:
                    credit_score = 900 + (rank - 0.5506) / (1 - 0.5506) * 100
                elif rank >= 0.3134:
                    credit_score = 800 + (rank - 0.3134) / (0.5506 - 0.3134) * 100
                elif rank >= 0.0592:
                    credit_score = 700 + (rank - 0.0592) / (0.3134 - 0.0592) * 100
                elif rank >= 0.0438:
                    credit_score = 600 + (rank - 0.0438) / (0.0592 - 0.0438) * 100
                else:
                    credit_score = 300 + (rank / 0.0438) * 300

                # 신용 등급 계산
                if credit_score >= 900:
                    grade = '1등급'
                elif credit_score >= 800:
                    grade = '2등급'
                elif credit_score >= 700:
                    grade = '3등급'
                elif credit_score >= 600:
                    grade = '4등급'
                else:
                    grade = '5등급'

                results.append({
                    "user_id": int(df.loc[idx, 'user_id']),
                    "credit_score": round(float(credit_score), 1),
                    "credit_grade": grade,
                    "percentile": round(float(percentile_ranks[idx] * 100), 2),
                    "logit_score": round(float(logit_score[idx]), 2)
                })

            return results

    except Exception as e:
        print(f"Error: {str(e)}")  # 디버깅용 출력
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-model")
async def check_model():
    try:
        return {
            "current_working_directory": os.getcwd(),
            "base_directory": str(BASE_DIR),
            "model_path": str(MODEL_PATH),
            "file_exists": os.path.exists(MODEL_PATH)
        }
    except Exception as e:
        return {"error": str(e)}