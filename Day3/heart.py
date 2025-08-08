import pandas as pd
import sklearn.neighbors as knn



dataset=pd.read_csv("heart.csv")

x=dataset[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]       
y=dataset[["target"]] 





model = knn.KNeighborsRegressor(n_neighbors=5)
model.fit(x,y)   

#----------------------------------
a1 = int(input("Enter Age:"))
a2 = int(input("Enter Sex:"))
a3 = int(input("Enter Cp:"))
a4 = int(input("Enter Trestbps:"))

a5 = int(input("Enter Chol:"))
a6 = int(input("Enter fbs:"))
a7 = int(input("Enter Rest ecg:"))
a8 = int(input("Enter Talach:"))
a9 = int(input("Enter Exang:"))
a10 = int(input("Enter Old peak:"))
a11 = int(input("Enter Slope:"))
a12 = int(input("Enter Ca:"))
a13 = int(input("Enter thal:"))

#----------------------------------
#eg value
#52,1,0,125,212,0,1,168,0,1,2,2,3
#a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13
print(model.predict([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13]]))