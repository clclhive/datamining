# 고객 이탈 예측을 위한 데이터 전처리 및 모델 코드 (수정 포함)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, classification_report, r2_score

# 데이터 불러오기
data = pd.read_csv('C:/Users/jwk72/Documents/DATA_MINING/bank-full.csv', delimiter=';')

# 데이터 확인
print(data.head())

# 범주형 변수 처리 - One-Hot Encoding
categorical_features = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# 스케일링 - age 변수는 원래 값을 유지
drop_features = ['age']
numerical_features = data.columns.difference(['y_yes', *drop_features])
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 데이터셋 나누기 (7:1:2 비율)
X = data.drop(['y_yes'], axis=1)
y = data['y_yes']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# 탐색적 데이터 분석 (EDA)
# 나이와 이탈 여부 시각화 (age 변수 원래 값을 사용)
plt.figure(figsize=(10, 6))
sns.histplot(x='age', hue='y_yes', data=data, multiple='stack', bins=30)
plt.title('Age Distribution by Churn')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 통화 시간(duration)과 이탈 여부 시각화
plt.figure(figsize=(10, 6))
sns.histplot(x='duration', hue='y_yes', data=data, multiple='stack', bins=30)
plt.title('Duration Distribution by Churn')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.show()

# 주택 대출 여부(housing_yes)와 이탈 여부 시각화
plt.figure(figsize=(10, 6))
sns.countplot(x='housing_yes', hue='y_yes', data=data)
plt.title('Housing Loan by Churn')
plt.xlabel('Housing Loan (Yes: 1, No: 0)')
plt.ylabel('Count')
plt.show()

# 상관관계 히트맵 (상관계수가 0.1 이상인 변수만 선택)
corr_matrix = data.corr()
important_features = corr_matrix.index[np.abs(corr_matrix["y_yes"]) > 0.1]
plt.figure(figsize=(12, 10))
sns.heatmap(data[important_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Filtered Important Features)')
plt.show()

# 선형 회귀 모델 구축 및 평가
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_val)

# 선형 회귀 평가
mse = mean_squared_error(y_val, y_pred_lr)
mae = mean_absolute_error(y_val, y_pred_lr)
print(f'MSE: {mse}, MAE: {mae}')

# 선형 회귀 가시화 및 해석
# 잔차 플롯
residuals = y_val - y_pred_lr
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lr, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

# 로지스틱 회귀 모델 구축 및 평가
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_val)

# 혼동 행렬 및 성능 지표 출력
conf_matrix = confusion_matrix(y_val, y_pred_logistic)
print(conf_matrix)
print(classification_report(y_val, y_pred_logistic))

# Shrinkage 모델 - Lasso, Ridge, ElasticNet 적용
# Lasso 모델
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_val)
print('Lasso R^2:', r2_score(y_val, lasso_pred))

# Ridge 모델
ridge = Ridge(alpha=0.01)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_val)
print('Ridge R^2:', r2_score(y_val, ridge_pred))

# ElasticNet 모델
elasticnet = ElasticNet(alpha=0.01, l1_ratio=0.5)
elasticnet.fit(X_train, y_train)
elasticnet_pred = elasticnet.predict(X_val)
print('ElasticNet R^2:', r2_score(y_val, elasticnet_pred))

# Shrinkage 모델 평가 요약 및 설명
print("\nShrinkage 모델 비교:")
print(f"Lasso R^2: {r2_score(y_val, lasso_pred)}")
print(f"Ridge R^2: {r2_score(y_val, ridge_pred)}")
print(f"ElasticNet R^2: {r2_score(y_val, elasticnet_pred)}")
