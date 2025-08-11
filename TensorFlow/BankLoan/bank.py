import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import r2_score,mean_squared_error
import math 


mydata=pd.read_csv("scenario_8_bank_loans_100k.csv")






x=mydata[["Age","Income","Credit_Score","Loan_Amount"]]     
y=mydata[["Approved"]]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = Sequential()

model.add(Dense(10,activation="relu",input_shape=(4,))) #no of inputs 
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")
model.fit(x_train,y_train,epochs=100)

y_pred = model.predict(x_test)
print("mse= ", mean_squared_error(y_test,y_pred))
print("rmse= ",math.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 score= ",r2_score(y_test,y_pred))


joblib.dump(model,"bank_model.pkl")

print("Code executed")
