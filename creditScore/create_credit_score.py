from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# DB 연결 설정
username = ""
password = ""
engine = create_engine(f"mysql+pymysql://{username}:{password}@regularmark.iptime.org:{port}/web_test")
Session = sessionmaker(bind=engine)

# DB에서 데이터 가져오기
with Session() as session:
    df = pd.read_sql_query("SELECT * FROM web_test.credit_score_evaluation", session.connection())
    
    # 컬럼명을 대문자로 변경
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
    model_df['quarter_seq'] = df['quarter_seq']  # quarter_seq는 그대로 사용
    
    # 신용 점수 예측 함수
    def predict_credit_score(data, model_path='credit_model.pkl'):
        try:
            # 모델 로드
            with open(model_path, 'rb') as f:
                model_info = pickle.load(f)

            coefficients = model_info['coefficients']
            scaler = model_info['scaler']
            required_features = model_info['required_features']

            # 데이터 스케일링
            X = data[required_features]
            X_scaled = scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=required_features, index=data.index)

            # 상수항 추가
            X_scaled.insert(0, 'const', 1)

            # 로짓 점수 계산
            logit_score = pd.Series(np.dot(X_scaled, coefficients), index=data.index)
            percentile_ranks = logit_score.rank(pct=True)

            # 신용 점수 계산
            credit_scores = pd.Series(index=data.index)

            for idx in data.index:
                rank = percentile_ranks[idx]
                if rank >= 0.5506:
                    credit_scores[idx] = 900 + (rank - 0.5506) / (1 - 0.5506) * 100
                elif rank >= 0.3134:
                    credit_scores[idx] = 800 + (rank - 0.3134) / (0.5506 - 0.3134) * 100
                elif rank >= 0.0592:
                    credit_scores[idx] = 700 + (rank - 0.0592) / (0.3134 - 0.0592) * 100
                elif rank >= 0.0438:
                    credit_scores[idx] = 600 + (rank - 0.0438) / (0.0592 - 0.0438) * 100
                else:
                    credit_scores[idx] = 300 + (rank / 0.0438) * 300

            # 신용 등급 부여
            def get_grade(score):
                if score >= 900: return '1등급'
                elif score >= 800: return '2등급'
                elif score >= 700: return '3등급'
                elif score >= 600: return '4등급'
                else: return '5등급'

            credit_grades = credit_scores.apply(get_grade)

            results = pd.DataFrame({
                'credit_score': credit_scores.round(1),
                'credit_grade': credit_grades,
                'percentile': (percentile_ranks * 100).round(2),
                'logit_score': logit_score.round(2)
            })

            return results

        except Exception as e:
            print(f"예측 중 오류 발생: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    # 예측 실행
    predictions = predict_credit_score(model_df)
    
    # 결과 출력
    if predictions is not None:
        for idx in predictions.index:
            print(f"고객id: {df.loc[idx, 'user_id']} - "
                  f"신용점수: {predictions.loc[idx, 'credit_score']:.1f} - "
                  f"신용등급: {predictions.loc[idx, 'credit_grade']} - "
                  f"상위 {predictions.loc[idx, 'percentile']:.1f}% - "
                  f"로짓점수: {predictions.loc[idx, 'logit_score']:.1f}")
