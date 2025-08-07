import pandas as pd
import sklearn.neighbors as knn
import matplotlib.pyplot as plot


dataset=pd.read_csv("gym_vs_weight_loss.csv")

x=dataset[["weekly_gym_hours"]]       
y=dataset[["weight_loss_kg"]] 


plot.scatter(x,y)    
plot.show()


model = knn.KNeighborsRegressor(n_neighbors=5)
model.fit(x,y)                 

print(model.predict([[4.5]])) 