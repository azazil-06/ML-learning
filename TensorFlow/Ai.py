import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


mydata=pd.read_csv("persons.csv")

gender_le= LabelEncoder()
bodytype_le= LabelEncoder()

mydata["gender_encoded"] = gender_le.fit_transform(mydata[["Gender"]])
mydata["body_encoded"] = bodytype_le.fit_transform(mydata[["BodyType"]])



x=mydata[["Age","gender_encoded","body_encoded","Height"]]     
y=mydata[["Weight"]]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = Sequential()

model.add(Dense(10,activation="relu",input_shape=(4,)))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")
model.fit(x_train,y_train,epochs=100)


joblib.dump(model,"Ai_model.pkl")

print("Code executed")