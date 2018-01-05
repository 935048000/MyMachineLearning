# Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# 函数来获取数据,解析csv文件
def get_data(file_name):
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    for single_square_feet ,single_price_value in zip(data['square_feet'],data['price']):
       X_parameter.append([float(single_square_feet)])
       Y_parameter.append(float(single_price_value))
    return X_parameter,Y_parameter


# 把X_parameter和Y_parameter拟合为线性回归模型
# 将数据与线性模型相适应的函数
def linear_model_main(X_parameters, Y_parameters, predict_value):
    # 创建线性回归对象
    regr = linear_model.LinearRegression ()
    regr.fit (X_parameters, Y_parameters)
    predict_outcome = regr.predict (predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions


# 要预测的平方英尺值为700
X,Y = get_data('./input_data.csv')
predictvalue = 700
result = linear_model_main(X,Y,predictvalue)
print ("截距值 " , result['intercept'])
print ("系数" , result['coefficient'])
print ("预测值: ",result['predicted_value'])
print("\n\n",X,Y)

# 输入为X_parameters和Y_parameters，显示出数据拟合的直线。
def show_linear_line(X_parameters,Y_parameters):
     # 创建线性回归对象
     regr = linear_model.LinearRegression()
     regr.fit(X_parameters, Y_parameters)
     plt.scatter(X_parameters,Y_parameters,color='blue')
     plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
     plt.xticks(())
     plt.yticks(())
     plt.show()

# show_linear_line(X,Y)