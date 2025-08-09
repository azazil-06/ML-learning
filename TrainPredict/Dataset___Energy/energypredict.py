import joblib
model = joblib.load("energy_model.pkl")
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


m=le.fit_transform(["Commercial"])[0]
n=le.fit_transform(["Weekday"])[0]

y= model.predict([[m,29187,82,39,23.54,n]])

#4991.64



print(y)