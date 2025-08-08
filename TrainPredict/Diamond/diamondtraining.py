import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import math 
from sklearn.preprocessing import LabelEncoder
import joblib


mydata=pd.read_csv("diamonds.csv")

le = LabelEncoder()
lg = LabelEncoder()
lk = LabelEncoder()

mydata["Cut_encoded"] = le.fit_transform(mydata[["cut"]])
mydata["Colour_encoded"] = lg.fit_transform(mydata[["color"]])
mydata["Clarity_encoded"] = lk.fit_transform(mydata[["clarity"]])

# Drop rows with any NaN values
mydata = mydata.dropna()

x = mydata[["carat","depth","table","x","y","z"]]
y = mydata[["price"]]




model = lm.LinearRegression()
model.fit(x,y)

joblib.dump(model,"diamonds_model.pkl")




