import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plot


dataset=pd.read_csv("study_hours_vs_scores.csv")

x=dataset[["no_of_hours"]]       
y=dataset[["score"]] 


plot.scatter(x,y)    
plot.show()


model=lm.LinearRegression()
model.fit(x,y)                 

print(model.predict([[8]])) 