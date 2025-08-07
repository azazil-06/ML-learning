import pandas as pd
import sklearn.neighbors as knn
import matplotlib.pyplot as plot


dataset=pd.read_csv("salary.csv")

x=dataset[["years_of_experience"]]       
y=dataset[["salary"]] 


plot.scatter(x,y)    
plot.show()


model = knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)                 

print(model.predict([[160]])) 