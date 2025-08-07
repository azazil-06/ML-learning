import pandas as pd
import sklearn.neighbors as knn
import matplotlib.pyplot as plot


dataset=pd.read_csv("data_height.csv")

x=dataset[["height"]]       
y=dataset[["weight"]] 


plot.scatter(x,y)    
plot.show()


model = knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)                 

print(model.predict([[160]])) 