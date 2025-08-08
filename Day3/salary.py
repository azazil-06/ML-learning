import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plot
from sklearn.metrics import r2_score,mean_squared_error
import math
from sklearn.model_selection import train_test_split



dataset=pd.read_csv("salary.csv")



x=dataset[["years_of_experience"]]       
y=dataset[["salary"]] 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)



model=lm.LinearRegression()
model.fit(x_train,y_train)   

y_pred = model.predict(x_test)
print(dataset)
print(y_pred)

print(model.predict([[4.5]])) 

print("mse= ", mean_squared_error(y_test,y_pred))
print("rmse= ",math.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 score= ",r2_score(y_test,y_pred))

























