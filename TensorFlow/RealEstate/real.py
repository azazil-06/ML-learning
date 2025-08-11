import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import r2_score,mean_squared_error
import math 


mydata=pd.read_csv("real_estate_100k.csv")

season_le = LabelEncoder()


mydata["fur_encoded"] = season_le.fit_transform(mydata[["Furnishing"]])




x=mydata[["Area_sqft","Bedrooms","Bathrooms","City_Tier","fur_encoded"]]     
y=mydata[["Price"]]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = Sequential()

model.add(Dense(10,activation="relu",input_shape=(5,)))  
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")
model.fit(x_train,y_train,epochs=50)

y_pred = model.predict(x_test)
print("mse= ", mean_squared_error(y_test,y_pred))
print("rmse= ",math.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 score= ",r2_score(y_test,y_pred))


joblib.dump(model,"realestate_model.pkl")

print("Code executed")