import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import math 
from sklearn.preprocessing import LabelEncoder
import joblib


mydata1=pd.read_csv("train_energy_data.csv")

le = LabelEncoder()



mydata1["Build_encoded"] = le.fit_transform(mydata1[["Building Type"]])
mydata1["DOW_encoded"] = le.fit_transform(mydata1[["Day of Week"]])



mydata = mydata1.dropna()

x = mydata1[["Build_encoded","Square Footage","Number of Occupants","Appliances Used","Average Temperature","DOW_encoded"]]
y = mydata1[["Energy Consumption"]]




model = lm.LinearRegression()
model.fit(x,y)

joblib.dump(model,"energy_model.pkl")
print("Model created")

#---------------------------ACCURACY
mydata2=pd.read_csv("test_energy_data.csv")
le = LabelEncoder()
lg = LabelEncoder()
mydata2["Build_encoded"] = le.fit_transform(mydata2[["Building Type"]])
mydata2["DOW_encoded"] = le.fit_transform(mydata2[["Day of Week"]])
x_test = mydata2[["Build_encoded","Square Footage","Number of Occupants","Appliances Used","Average Temperature","DOW_encoded"]]
y_test= mydata2[["Energy Consumption"]]
y_pred = model.predict(x_test)
print("mse= ", mean_squared_error(y_test,y_pred))
print("rmse= ",math.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 score= ",r2_score(y_test,y_pred))
#----------------------------------------------------------------------



print("Code ran successfully" )
