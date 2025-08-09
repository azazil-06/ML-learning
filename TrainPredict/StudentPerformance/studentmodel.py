import pandas as pd

import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import math 
from sklearn.preprocessing import LabelEncoder
import joblib




mydata=pd.read_csv("Student_Performance.csv")

le = LabelEncoder()



mydata["EA_encoded"] = le.fit_transform(mydata[["Extracurricular Activities"]])




mydata = mydata.dropna()

x = mydata[["Hours Studied","Previous Scores","EA_encoded","Sleep Hours","Sample Question Papers Practiced"]]
y = mydata[["Performance Index"]]



x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)



model = lm.LinearRegression()
model.fit(x_train,y_train)

joblib.dump(model,"student_model.pkl")


y_pred = model.predict(x_test)
print("mse= ", mean_squared_error(y_test,y_pred))
print("rmse= ",math.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 score= ",r2_score(y_test,y_pred))



print("Code ran successfully" )
