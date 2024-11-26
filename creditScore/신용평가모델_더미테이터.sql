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


INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (2, 749401, 0.45, 726609, 0, 98078, 85999, 124936, 0.21, 52146, 0, 74574, 208716, 5862852, 1.42, 133735463, 1);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (2, 645346, 0.51, 860416, 0, 93474, 91813, 125610, 0.22, 55329, 0, 77618, 203044, 6050356, 1.39, 132016553, 2);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (1, 677241, 0.54, 804399, 0, 105824, 91155, 132965, 0.2, 56627, 0, 81039, 199426, 5485959, 1.26, 152603465, 3);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (1, 738402, 0.51, 720187, 0, 107297, 98815, 131196, 0.21, 53464, 0, 76164, 189471, 5596139, 1.28, 127113088, 4);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (1, 670736, 0.54, 867060, 0, 101118, 89894, 137895, 0.2, 53194, 0, 79393, 196184, 5427998, 1.29, 146097687, 5);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (1, 717071, 0.46, 764637, 0, 95473, 84437, 121938, 0.19, 54287, 0, 70484, 206375, 5556692, 1.4, 128428237, 6);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (1, 692550, 0.49, 750257, 0, 95616, 89318, 117821, 0.19, 58147, 0, 68786, 201637, 6576641, 1.34, 131072534, 7);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (2, 731311, 0.5, 764442, 0, 97234, 96212, 141499, 0.22, 54844, 0, 73120, 188681, 5770752, 1.2, 145818137, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (2, 514558, 0.36, 613208, 9416, 79606, 61299, 76843, 0.28, 43554, 0, 52146, 192620, 8072545, 1.65, 103664664, 1);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (3, 522273, 0.43, 592921, 10622, 84213, 62856, 69598, 0.3, 39541, 1, 53530, 217727, 8289293, 1.42, 109343967, 2);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (2, 524053, 0.42, 657252, 9273, 87816, 62348, 74936, 0.33, 40557, 1, 45021, 187616, 8678871, 1.42, 100508013, 3);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (3, 496344, 0.39, 579579, 10787, 83244, 60354, 66549, 0.32, 42291, 0, 47218, 213205, 8627875, 1.42, 100516892, 4);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (2, 506339, 0.43, 573195, 10062, 87540, 62076, 64196, 0.3, 36993, 1, 49109, 196141, 7741898, 1.46, 105938031, 5);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (3, 528029, 0.4, 581777, 10086, 86278, 56583, 72688, 0.33, 39436, 1, 49311, 210549, 8271187, 1.5, 90920543, 6);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (2, 509991, 0.36, 602854, 10111, 84763, 54523, 64497, 0.3, 42420, 1, 53838, 213647, 8741886, 1.51, 102407048, 7);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (2, 468500, 0.44, 615423, 10467, 75228, 62974, 73404, 0.33, 41972, 1, 53269, 215637, 8497885, 1.55, 101532134, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (3, 415975, 0.45, 521373, 20107, 66628, 52983, 59844, 0.4, 37770, 1, 41504, 204755, 9300712, 1.6, 89427623, 1);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (4, 409745, 0.46, 570207, 21676, 63071, 47574, 54434, 0.4, 34025, 0, 47962, 209721, 9149796, 1.45, 83268083, 2);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (3, 367218, 0.41, 597644, 20946, 68560, 49429, 57600, 0.37, 38463, 1, 41991, 187581, 9344065, 1.57, 87429547, 3);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (4, 396334, 0.44, 567012, 19327, 76225, 45983, 57655, 0.37, 34863, 0, 45288, 208280, 10777235, 1.63, 93187571, 4);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (3, 423654, 0.41, 595125, 18586, 66358, 54100, 65542, 0.38, 33296, 0, 46065, 200465, 10363557, 1.57, 98742431, 5);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (3, 370237, 0.46, 572593, 21028, 74700, 52064, 59352, 0.41, 33157, 0, 40585, 185982, 9278438, 1.67, 84844926, 6);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (3, 438589, 0.46, 586349, 18618, 76976, 45217, 54138, 0.36, 37016, 0, 40515, 187379, 9637572, 1.53, 93218539, 7);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (3, 402333, 0.43, 596457, 18312, 63387, 48130, 54851, 0.4, 35559, 1, 47943, 187562, 10036142, 1.48, 92514825, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (4, 308392, 0.51, 410649, 45316, 54860, 39151, 54276, 0.48, 30563, 1, 43765, 204524, 16006064, 1.84, 75821243, 1);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (4, 290066, 0.49, 398424, 48206, 65896, 38214, 49198, 0.5, 28675, 1, 42486, 192662, 16105744, 1.95, 76825298, 2);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (5, 298838, 0.53, 379213, 54760, 59698, 38496, 46309, 0.5, 28639, 0, 42021, 213952, 13618689, 1.87, 64692926, 3);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (5, 308311, 0.51, 376162, 46782, 54788, 39292, 49398, 0.5, 31598, 1, 43147, 192207, 15780451, 1.81, 74642046, 4);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (5, 289420, 0.51, 432332, 49787, 65841, 41964, 45784, 0.52, 28895, 0, 38219, 192384, 16006569, 1.9, 74155054, 5);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (5, 282170, 0.45, 430658, 48016, 54946, 40091, 53426, 0.55, 31557, 1, 39968, 216798, 15521313, 1.86, 66490727, 6);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (4, 278751, 0.5, 417782, 45885, 59769, 36715, 48377, 0.53, 27605, 1, 39457, 206081, 14821222, 1.9, 74137211, 7);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (4, 290262, 0.53, 405538, 52913, 58153, 37536, 48261, 0.53, 32730, 0, 40686, 189674, 13870525, 1.72, 64359800, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (6, 242696, 0.5, 335265, 61380, 48848, 33417, 41403, 0.59, 25745, 0, 35644, 212979, 16975067, 2.15, 65892481, 1);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (5, 271425, 0.59, 318090, 61217, 47771, 36858, 46052, 0.61, 27183, 0, 34494, 217801, 19434616, 1.81, 65876895, 2);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (6, 261309, 0.52, 331910, 57930, 48882, 34337, 47851, 0.6, 25238, 1, 36069, 206528, 18434870, 1.93, 58176204, 3);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (6, 252797, 0.55, 346610, 62586, 52357, 33768, 41806, 0.55, 25418, 0, 35362, 190210, 18632847, 2.13, 61511694, 4);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (6, 226557, 0.59, 338428, 64395, 46322, 35726, 42528, 0.55, 25623, 0, 35352, 209158, 17024616, 1.98, 61179438, 5);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (6, 241968, 0.53, 322330, 58366, 45424, 32107, 41831, 0.61, 27297, 1, 37200, 218334, 18779723, 1.96, 56666584, 6);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (5, 263391, 0.5, 335712, 61008, 45218, 36442, 47362, 0.63, 27035, 0, 35343, 186814, 17927093, 1.99, 57835323, 7);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (5, 246962, 0.51, 332927, 56747, 46415, 37562, 43939, 0.57, 23094, 0, 36782, 196816, 18466394, 1.83, 64235587, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (7, 185473, 0.7, 285006, 97125, 40680, 32060, 35226, 0.86, 18617, 1, 29665, 201625, 26422118, 2.58, 38976440, 1);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (7, 195083, 0.64, 328818, 98213, 40394, 31596, 37301, 0.82, 19598, 1, 30224, 201741, 26516582, 2.4, 39285770, 2);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (6, 204840, 0.73, 317146, 108122, 43443, 30139, 33449, 0.86, 21655, 2, 31443, 192307, 23047440, 2.4, 36314147, 3);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (6, 195494, 0.63, 306015, 108011, 43375, 30449, 36142, 0.86, 21669, 1, 31898, 185498, 25747847, 2.39, 40413841, 4);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (6, 213145, 0.75, 290325, 109174, 43600, 29841, 37450, 0.78, 18000, 1, 28323, 188740, 24561280, 2.51, 43581541, 5);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (6, 201552, 0.63, 311312, 90940, 38341, 31557, 34336, 0.78, 18305, 1, 27251, 219232, 24098262, 2.59, 43282252, 6);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (6, 212262, 0.66, 276621, 103415, 42208, 27613, 36641, 0.84, 18594, 1, 31559, 205337, 22607759, 2.32, 41731270, 7);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (7, 200902, 0.77, 323478, 100279, 42443, 28298, 33368, 0.88, 18517, 2, 28051, 221635, 25143734, 2.48, 40168626, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (7, 166943, 0.72, 249820, 116427, 36635, 24678, 28584, 0.89, 19720, 1, 25073, 206643, 31196366, 3.05, 38406112, 1);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (7, 186336, 0.75, 227619, 124943, 34836, 27154, 27993, 0.86, 18042, 2, 26061, 197229, 32702306, 2.79, 36863417, 2);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (8, 192998, 0.74, 270927, 111308, 36819, 22901, 29031, 0.77, 16435, 1, 24750, 218163, 32665580, 2.72, 34278321, 3);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (7, 172825, 0.75, 254692, 118628, 34860, 22683, 31958, 0.8, 17097, 2, 27326, 217924, 28577542, 2.87, 35084157, 4);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (7, 192428, 0.79, 231577, 128689, 33840, 22542, 30119, 0.78, 18354, 1, 22512, 197164, 27719359, 2.88, 32095569, 5);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (8, 189896, 0.72, 251218, 123021, 38209, 24273, 30352, 0.88, 19466, 2, 25517, 212445, 27842401, 2.68, 37769257, 6);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (7, 196938, 0.71, 231493, 126558, 34468, 26315, 32023, 0.79, 17551, 1, 26313, 187462, 30507460, 2.94, 33162652, 7);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (8, 166699, 0.82, 268341, 128268, 38396, 23183, 31020, 0.91, 16892, 2, 23189, 210472, 32435684, 3.06, 36309423, 8);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (8, 164651, 0.88, 183399, 159815, 30480, 21269, 25218, 1.04, 16236, 1, 20643, 205761, 38068279, 3.2, 25989291, 1);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (8, 140791, 0.9, 213195, 159084, 32548, 18792, 26005, 0.86, 16069, 1, 21899, 206687, 33709166, 3.19, 22649591, 2);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (8, 155100, 0.97, 214193, 143009, 32515, 20746, 23005, 0.99, 15626, 1, 18128, 215465, 33873715, 3.26, 26196144, 3);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (9, 147690, 0.83, 198608, 164967, 32711, 20912, 24858, 0.96, 15675, 2, 18426, 193374, 37772717, 3.08, 23543849, 4);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (9, 158180, 0.97, 217552, 142143, 28312, 20756, 27355, 0.93, 14386, 1, 21893, 185673, 35704576, 3.05, 27195094, 5);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (8, 142617, 0.84, 184553, 154954, 30511, 21966, 23064, 1.03, 14345, 2, 19585, 217882, 38248740, 3.32, 26103370, 6);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (9, 137358, 0.93, 186973, 153732, 31450, 18535, 26100, 1.03, 14251, 2, 21293, 204884, 32888031, 3.48, 24253355, 7);
INSERT INTO credit_score_evaluation (user_id, tot_use_am, ues_income, crdsl_use_am, cnf_use_am, plsanit_am, fsbz_am, trvlec_am, dan_rt, dist_am, life_stage_dan, clothgds_am, att_ym, debt, debt_ratio, income, quarter_seq)
        VALUES (8, 143171, 0.96, 199019, 155649, 29249, 20082, 26211, 1.02, 15570, 1, 19396, 204239, 31908903, 3.45, 27311825, 8);
