import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as knn
from sklearn.preprocessing import LabelEncoder

mydata = pd.read_csv("big_salary_data.csv")

le = LabelEncoder()
mydata["education_qualification_encoded"] = le.fit_transform(mydata[["education_qualification"]]) #x inte string value encoded into number

x = mydata[["education_qualification_encoded"]]
y = mydata[["salary"]]

model = knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)                 

print(model.predict([[2]])) 

print(mydata)