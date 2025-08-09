import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import math

from sklearn.preprocessing import LabelEncoder
import joblib

mydata = pd.read_csv("Fish.csv")

le = LabelEncoder()
mydata["Species_encoded"] = le.fit_transform(mydata[["Species"]])

x = mydata[["Weight", "Length1", "Length2", "Length3", "Height", "Width"]]
y = mydata[["Species_encoded"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = lm.LinearRegression()
model.fit(x_train, y_train)

joblib.dump(model, "fish_model.pkl")

y_pred = model.predict(x_test)
print("mse= ", mean_squared_error(y_test, y_pred))
print("rmse= ", math.sqrt(mean_squared_error(y_test, y_pred)))
print("r2 score= ", r2_score(y_test, y_pred))

print("Code ran successfully")

