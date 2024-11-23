import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import shap
import xgboost as xgb
from lime import lime_tabular
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
df = pd.read_csv('Derived_Variables.csv')

def prepare_data(df):
    """데이터 전처리 함수"""
    # 숫자형 컬럼만 선택
    numeric_df = df.select_dtypes(include=[np.number])

    # 결측치 처리
    numeric_df = numeric_df.fillna(numeric_df.mean())

    # 타겟 변수 설정 (예: TOT_USE_AM)
    target = 'TOT_USE_AM'
    features = [col for col in numeric_df.columns if col != target]

    X = numeric_df[features]
    y = numeric_df[target]

    return X, y, features

def random_forest_importance(X, y, features):
    """1. Random Forest MDI 중요도"""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title('Random Forest Feature Importance (MDI)')
    plt.tight_layout()
    plt.show()

    return importance_df

def shap_importance(X, y):
    """2. SHAP 중요도"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.tight_layout()

    return shap_values

def permutation_imp(X, y, features):
    """3. Permutation Importance"""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

    perm_imp_df = pd.DataFrame({
        'feature': features,
        'importance': result.importances_mean
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=perm_imp_df.head(20), x='importance', y='feature')
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
    plt.show()

    return perm_imp_df

def gradient_boosting_importance(X, y, features):
    """4. Gradient Boosting 중요도"""
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X, y)

    xgb_imp_df = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=xgb_imp_df.head(20), x='importance', y='feature')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.show()

    return xgb_imp_df

def pca_analysis(X):
    """6. PCA 분석"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    # 설명된 분산 비율
    explained_variance_ratio = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'explained_variance_ratio': pca.explained_variance_ratio_
    })

    # 주성분에 대한 변수 기여도
    components_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=X.columns
    )

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.tight_layout()
    plt.show()

    return explained_variance_ratio, components_df

def spearman_correlation(X):
    """7. 스피어만 상관계수"""
    corr = X.corr(method='spearman')

    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, cmap='RdBu_r', center=0)
    plt.title('Spearman Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # 높은 상관관계를 가진 변수쌍 찾기
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) >= 0.7:  # 임계값 0.7
                high_corr.append({
                    'var1': corr.columns[i],
                    'var2': corr.columns[j],
                    'correlation': corr.iloc[i, j]
                })

    high_corr_df = pd.DataFrame(high_corr).sort_values('correlation',
                                                      key=abs,
                                                      ascending=False)

    return high_corr_df

# 메인 실행 함수
def main():
    X, y, features = prepare_data(df)

    print("1. Random Forest MDI 중요도 분석")
    rf_imp = random_forest_importance(X, y, features)
    print("\nTop 10 features by RF importance:")
    print(rf_imp.head(10))

    print("\n2. SHAP 중요도 분석")
    shap_values = shap_importance(X, y)

    print("\n3. Permutation Importance 분석")
    perm_imp = permutation_imp(X, y, features)
    print("\nTop 10 features by permutation importance:")
    print(perm_imp.head(10))

    print("\n4. Gradient Boosting 중요도 분석")
    xgb_imp = gradient_boosting_importance(X, y, features)
    print("\nTop 10 features by XGBoost importance:")
    print(xgb_imp.head(10))

    print("\n6. PCA 분석")
    explained_var, components = pca_analysis(X)
    print("\nExplained variance ratio by first 5 components:")
    print(explained_var.head())

    print("\n7. 스피어만 상관계수 분석")
    high_corr = spearman_correlation(X)
    print("\nTop 10 highly correlated variable pairs:")
    print(high_corr.head(10))

if __name__ == "__main__":
    main()
