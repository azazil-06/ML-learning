import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import math 


mydata=pd.read_csv("learn.csv")
x = mydata[["height"]]
y = mydata[["weight"]]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)


pt.scatter(x_train,y_train)
pt.show()
model = lm.LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("mse= ", mean_squared_error(y_test,y_pred))
print("rmse= ",math.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 score= ",r2_score(y_test,y_pred))

print("coefficient:",model.coef_[0])
print("intercept:",model.intercept_[0])
print(model.predict([[160]]))