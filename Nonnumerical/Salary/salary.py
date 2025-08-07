import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

mydata = pd.read_csv("salary_data.csv")

le = LabelEncoder()
mydata["education_qualification_encoded"] = le.fit_transform(mydata[["education_qualification"]]) #x inte string value encoded into number

x = mydata[["education_qualification_encoded"]]
y = mydata[["salary"]]

model = LinearRegression()
model.fit(x, y)

print("coefficient:", model.coef_[0])
print("intercept:", model.intercept_[0])
print(model.predict([[0]]))

print(mydata)