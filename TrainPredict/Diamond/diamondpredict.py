import joblib
model = joblib.load("diamonds_model.pkl")
from sklearn.preprocessing import LabelEncoder



y= model.predict([[0.23,61.5,55,3.95,3.98,2.43]])





print(y)