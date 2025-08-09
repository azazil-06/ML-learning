import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import math 
from sklearn.preprocessing import LabelEncoder


mydata=pd.read_csv("titanic.csv")

le = LabelEncoder()
lg = LabelEncoder()
lk = LabelEncoder()

mydata["Cabin_encoded"] = le.fit_transform(mydata[["Cabin"]])
mydata["Embarked_encoded"] = lg.fit_transform(mydata[["Embarked"]])
mydata["Sex_encoded"] = lk.fit_transform(mydata[["Sex"]])

# Drop rows with any NaN values
mydata = mydata.dropna()

x = mydata[["Pclass","Sex_encoded","Age","SibSp","Parch","Cabin_encoded","Embarked_encoded"]]
y = mydata[["Survived"]]


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

model = lm.LinearRegression()
model.fit(x_train,y_train)


y_pred = model.predict(x_test)
print("mse= ", mean_squared_error(y_test,y_pred))
print("rmse= ",math.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 score= ",r2_score(y_test,y_pred))

print("coefficient:",model.coef_[0])
print("intercept:",model.intercept_[0])

"""
m=le.fit_transform(["male"])[0]  #array 1st/sngl string index pass cheyyan use this
n=lg.fit_transform(["C24"])[0]
bk=lk.fit_transform(["S"])[0]

y= model.predict([[3,m,22,1,0,n,bk]])


""" 

m=le.fit_transform(["female"])[0]  
n=lg.fit_transform(["C0"])[0]
bk=lk.fit_transform(["S"])[0]

y= model.predict([[3,m,90,0,2,n,bk]])





if y>0.5 :
    print("This person survived")

else:
    print("This person did not survive")