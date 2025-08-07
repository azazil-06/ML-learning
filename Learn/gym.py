import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plot


dataset=pd.read_csv("gym_vs_weight_loss.csv")

x=dataset[["weekly_gym_hours"]]       
y=dataset[["weight_loss_kg"]] 


plot.scatter(x,y)    
plot.show()


model=lm.LinearRegression()
model.fit(x,y)                 

print(model.predict([[4]]))