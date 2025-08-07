import pandas as pd
import sklearn.linear_model as lm

dataset = pd.read_csv("weight_predict_real.csv")

x=dataset[["height","age","bmi","muscle_mass","body_fat"]]       
y=dataset[["weight"]] 


model=lm.LinearRegression()
model.fit(x,y)  

print(model.predict([[160,20,80,10,9]]))