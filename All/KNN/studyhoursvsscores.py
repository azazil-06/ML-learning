import pandas as pd
import sklearn.neighbors as knn
import matplotlib.pyplot as plot


dataset=pd.read_csv("study_hours_vs_scores.csv")

x=dataset[["no_of_hours"]]       
y=dataset[["score"]] 


plot.scatter(x,y)    
plot.show()


model = knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)                 

print(model.predict([[160]]))