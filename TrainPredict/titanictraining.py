import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import math 
from sklearn.preprocessing import LabelEncoder
import joblib


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


joblib.dump(model,"titanic_model.pkl")

