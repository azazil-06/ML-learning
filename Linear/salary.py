import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plot


dataset=pd.read_csv("salary.csv")

x=dataset[["years_of_experience"]]       
y=dataset[["salary"]] 


plot.scatter(x,y)    
plot.show()


model=lm.LinearRegression()
model.fit(x,y)                 

print(model.predict([[4.5]])) 