CREATE TABLE credit_score_evaluation (
    user_id INT NOT NULL,                        -- 유저 ID
    idx INT NOT NULL AUTO_INCREMENT,            -- 자동 증가 인덱스
    tot_use_am BIGINT NOT NULL,                 -- 총 사용 금액
    ues_income DECIMAL(5,2) NOT NULL,           -- 소득 대비 사용 비율
    crdsl_use_am BIGINT NOT NULL,               -- 신용대출 사용 금액
    cnf_use_am BIGINT NOT NULL,                 -- 신용카드 사용 금액
    plsanit_am BIGINT NOT NULL,                 -- 공공요금
    fsbz_am BIGINT NOT NULL,                    -- 금융상품 금액
    trvlec_am BIGINT NOT NULL,                  -- 여행 지출 금액
    dan_rt DECIMAL(5,2) NOT NULL,               -- 위험도
    dist_am BIGINT NOT NULL,                    -- 배달비
    life_stage_dan TINYINT NOT NULL,            -- 생애주기 위험도
    clothgds_am BIGINT NOT NULL,                -- 의류 소비 금액
    att_ym INT NOT NULL,                        -- 연월
    debt BIGINT NOT NULL,                       -- 부채 금액
    debt_ratio DECIMAL(5,2) NOT NULL,           -- 부채 비율
    income BIGINT NOT NULL,                     -- 소득
    quarter_seq INT NOT NULL,                   -- 분기 시퀀스
    PRIMARY KEY (idx)                           -- 기본 키는 idx
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 1~3등급 데이터
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq) VALUES (1, 800000, 0.6, 900000, 0, 120000, 100000, 150000, 0.1, 60000, 0, 80000, 202312, 5000000, 1.2, 150000000, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq) VALUES (2, 700000, 0.5, 800000, 0, 100000, 90000, 130000, 0.2, 55000, 0, 75000, 202312, 6000000, 1.3, 140000000, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq) VALUES (3, 500000, 0.4, 600000, 10000, 80000, 60000, 70000, 0.3, 40000, 1, 50000, 202312, 8000000, 1.5, 100000000, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq) VALUES (4, 400000, 0.45, 550000, 20000, 70000, 50000, 60000, 0.4, 35000, 1, 45000, 202312, 10000000, 1.6, 90000000, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq) VALUES (5, 300000, 0.5, 400000, 50000, 60000, 40000, 50000, 0.5, 30000, 1, 40000, 202312, 15000000, 1.8, 70000000, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq) VALUES (6, 250000, 0.55, 350000, 60000, 50000, 35000, 45000, 0.6, 25000, 1, 35000, 202312, 18000000, 2.0, 60000000, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq) VALUES (7, 200000, 0.7, 300000, 100000, 40000, 30000, 35000, 0.8, 20000, 2, 30000, 202312, 25000000, 2.5, 40000000, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq) VALUES (8, 180000, 0.75, 250000, 120000, 35000, 25000, 30000, 0.85, 18000, 2, 25000, 202312, 30000000, 2.8, 35000000, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq) VALUES (9, 150000, 0.9, 200000, 150000, 30000, 20000, 25000, 0.95, 15000, 2, 20000, 202312, 35000000, 3.2, 25000000, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq) VALUES (10, 120000, 1.0, 180000, 200000, 25000, 15000, 20000, 0.99, 12000, 2, 15000, 202312, 40000000, 3.5, 20000000, 8);

-- 4등급 데이터
INSERT INTO credit_score_evaluation (user_id, TOT_USE_AM, UES_INCOME, CRDSL_USE_AM, CNF_USE_AM, PLSANIT_AM, FSBZ_AM, TRVLEC_AM, DAN_RT, DIST_AM, LIFE_STAGE_DAN, CLOTHGDS_AM, ATT_YM, DEBT, DEBT_RATIO, INCOME, quarter_seq) 
VALUES (11, 406639, 0.87, 434015, 60908, 55175, 43994, 50302, 0.65, 25592, 1, 40518, 207703, 22155264, 2.32, 44717566, 6);

INSERT INTO credit_score_evaluation (user_id, TOT_USE_AM, UES_INCOME, CRDSL_USE_AM, CNF_USE_AM, PLSANIT_AM, FSBZ_AM, TRVLEC_AM, DAN_RT, DIST_AM, LIFE_STAGE_DAN, CLOTHGDS_AM, ATT_YM, DEBT, DEBT_RATIO, INCOME, quarter_seq) 
VALUES (12, 368960, 0.64, 508325, 55912, 44072, 33146, 51439, 0.7, 30892, 1, 34912, 198414, 22555171, 1.95, 59255007, 7);

INSERT INTO credit_score_evaluation (user_id, TOT_USE_AM, UES_INCOME, CRDSL_USE_AM, CNF_USE_AM, PLSANIT_AM, FSBZ_AM, TRVLEC_AM, DAN_RT, DIST_AM, LIFE_STAGE_DAN, CLOTHGDS_AM, ATT_YM, DEBT, DEBT_RATIO, INCOME, quarter_seq) 
VALUES (13, 420418, 0.7, 441832, 54327, 57349, 36864, 39852, 0.89, 29074, 0, 31326, 192315, 17479744, 2.06, 49673295, 7);

INSERT INTO credit_score_evaluation (user_id, TOT_USE_AM, UES_INCOME, CRDSL_USE_AM, CNF_USE_AM, PLSANIT_AM, FSBZ_AM, TRVLEC_AM, DAN_RT, DIST_AM, LIFE_STAGE_DAN, CLOTHGDS_AM, ATT_YM, DEBT, DEBT_RATIO, INCOME, quarter_seq) 
VALUES (14, 360622, 0.71, 455487, 53501, 58306, 34840, 43386, 0.73, 28292, 1, 31996, 221441, 21821055, 2.58, 56162713, 8);

INSERT INTO credit_score_evaluation (user_id, TOT_USE_AM, UES_INCOME, CRDSL_USE_AM, CNF_USE_AM, PLSANIT_AM, FSBZ_AM, TRVLEC_AM, DAN_RT, DIST_AM, LIFE_STAGE_DAN, CLOTHGDS_AM, ATT_YM, DEBT, DEBT_RATIO, INCOME, quarter_seq) 
VALUES (15, 410885, 0.87, 580502, 56342, 48173, 37330, 37056, 0.72, 29732, 1, 41057, 226061, 21087765, 2.5, 59306030, 6);

-- 5등급 데이터
INSERT INTO credit_score_evaluation (user_id, TOT_USE_AM, UES_INCOME, CRDSL_USE_AM, CNF_USE_AM, PLSANIT_AM, FSBZ_AM, TRVLEC_AM, DAN_RT, DIST_AM, LIFE_STAGE_DAN, CLOTHGDS_AM, ATT_YM, DEBT, DEBT_RATIO, INCOME, quarter_seq) 
VALUES (16, 373052, 0.8, 554611, 151576, 50040, 29487, 24620, 0.77, 21964, 1, 28801, 152583, 34600928, 3.23, 48564885, 8);

INSERT INTO credit_score_evaluation (user_id, TOT_USE_AM, UES_INCOME, CRDSL_USE_AM, CNF_USE_AM, PLSANIT_AM, FSBZ_AM, TRVLEC_AM, DAN_RT, DIST_AM, LIFE_STAGE_DAN, CLOTHGDS_AM, ATT_YM, DEBT, DEBT_RATIO, INCOME, quarter_seq) 
VALUES (17, 269858, 1.13, 676202, 87992, 30586, 30337, 38110, 0.78, 25083, 1, 26402, 219575, 34425464, 3.22, 45545786, 7);

INSERT INTO credit_score_evaluation (user_id, TOT_USE_AM, UES_INCOME, CRDSL_USE_AM, CNF_USE_AM, PLSANIT_AM, FSBZ_AM, TRVLEC_AM, DAN_RT, DIST_AM, LIFE_STAGE_DAN, CLOTHGDS_AM, ATT_YM, DEBT, DEBT_RATIO, INCOME, quarter_seq) 
VALUES (18, 375869, 0.83, 685903, 95201, 49940, 26298, 22341, 1.06, 23099, 2, 24320, 236483, 23152799, 2.65, 30708907, 7);

INSERT INTO credit_score_evaluation (user_id, TOT_USE_AM, UES_INCOME, CRDSL_USE_AM, CNF_USE_AM, PLSANIT_AM, FSBZ_AM, TRVLEC_AM, DAN_RT, DIST_AM, LIFE_STAGE_DAN, CLOTHGDS_AM, ATT_YM, DEBT, DEBT_RATIO, INCOME, quarter_seq) 
VALUES (19, 274845, 0.71, 648985, 138701, 50520, 27827, 23898, 1.15, 14585, 2, 20915, 248512, 38401797, 2.23, 49392573, 5);

INSERT INTO credit_score_evaluation (user_id, TOT_USE_AM, UES_INCOME, CRDSL_USE_AM, CNF_USE_AM, PLSANIT_AM, FSBZ_AM, TRVLEC_AM, DAN_RT, DIST_AM, LIFE_STAGE_DAN, CLOTHGDS_AM, ATT_YM, DEBT, DEBT_RATIO, INCOME, quarter_seq) 
VALUES (20, 251402, 1.27, 680787, 119607, 48105, 27162, 24406, 1.18, 16021, 1, 28004, 246841, 36621747, 3.3, 31981881, 6);

