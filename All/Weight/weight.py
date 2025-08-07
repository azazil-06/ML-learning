import pandas as pd
import sklearn.linear_model as lm

dataset = pd.read_csv("weight_predict_real.csv")

x=dataset[["height","age","bmi","muscle_mass","body_fat"]]       
y=dataset[["weight"]] 

getHeight = int(input("Enter height:"))
getAge = int(input("Enter age:"))
getBMI = int(input("Enter BMI:"))
getMuscleMass = int(input("Enter musclemass:"))
getbodyfat = int(input("Enter bodyfat %:"))

model=lm.LinearRegression()
model.fit(x,y)  

print(model.predict([[getHeight,getAge,getBMI,getMuscleMass,getbodyfat]]))