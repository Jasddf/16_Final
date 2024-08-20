import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# 헤더
Header = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
          'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data = pd.read_csv('./data/3.housing.csv',
                   delim_whitespace=True, names=Header)

# 데이터 벗겨내기(정리)
array = data.values

# 독립변수 / 종속변수
X = array[:,0:13]
Y = array[:,13]
# print(data)

# 학습데이터 / 테스트데이터
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# scaler = MinMaxScaler(feature_range=(0,1))
# rescaled_X = scaler.fit_transform(X)
# print(rescaled_X)

model = LinearRegression()
model.fit(X_train,Y_train) # 모델 학습시키기
y_pred = model.predict(X_test) # 모델 학습시킨 값으로 종속변수 예측


#y_test 값 저장 및 Y 예측값 저장
# df_Y_test = pd.DataFrame(Y_test)
# df_Y_pred = pd.DataFrame(y_pred)
# df_Y_test.to_csv('./results/Y_test.csv')
# df_Y_pred.to_csv('./results/Y_pred.csv')

# 시각화 하기 비굣값 뒤에 [:OO]처럼 표 간소화 가능
plt.clf()
plt.scatter(range(len(X_test[:15])), Y_test[:15], color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(X_test[:15])), y_pred[:15], color='red', label='Predicted Values', marker='x')

plt.title('Housing Data')
plt.xlabel('DATA')
plt.ylabel('MEDV')
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')

kfold = KFold(n_splits=5)
mse = cross_val_score(model, X, Y, scoring='neg_mean_squared_error')
print(mse.mean())