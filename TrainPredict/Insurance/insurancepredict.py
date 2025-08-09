import joblib
model = joblib.load("insurance_model.pkl")
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

h=le.fit_transform(["male"])[0]
l=le.fit_transform(["no"])[0]
o=le.fit_transform(["northeast"])[0]

y= model.predict([[37,h,29.83,2,l,o]])
#6406.4107   


print(y)