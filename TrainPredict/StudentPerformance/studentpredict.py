import joblib
model = joblib.load("student_model.pkl")
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

m=le.fit_transform(["Yes"])[0]

y= model.predict([[7,73,m,5,6]])
#63.0
print(y)