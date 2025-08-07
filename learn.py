
#Linear regression

import pandas as pd #alias pd ,pandas to read dataset
import matplotlib.pyplot as plot
import sklearn.linear_model as lm

mydata=pd.read_csv("data_height.csv") #mydata ilott data_height.csv file kodukkan

x=mydata[["height"]]       #csv ile height table values x ilott
y=mydata[["weight"]]       #csv ile weight table values y ilott

plot.scatter(x,y)     #plot graph points
plot.show()

#training
model=lm.LinearRegression()
model.fit(x,y)                 #creating model y=mx+c


#predict
print(model.predict([[160]])) #passing values

#dataset print
print(mydata)


#y=mx+c de slope & c print
print("Coefficient", model.coef_[0])
print("Intercept", model.intercept_[0])


#Encoding string data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
mydata["education_qualification_encoded"] = le.fit_transform(mydata[["education_qualification"]]) #x inte string value encoded into number

x = mydata[["education_qualification_encoded"]]
y = mydata[["salary"]]



#-------------------------------------------------------------------------------------------------------


#Neighbors regressor

import pandas as pd
import sklearn.neighbors as knn
import matplotlib.pyplot as plot


dataset=pd.read_csv("data_height.csv")

x=dataset[["height"]]       
y=dataset[["weight"]] 


plot.scatter(x,y)    
plot.show()

#knn model ondakk
model = knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)                 

print(model.predict([[160]])) 


#--------------------------------------------------------------------------------------------------
#multiple values varuannel How?

dataset = pd.read_csv("weight_predict_real.csv")

x=dataset[["height","age","bmi","muscle_mass","body_fat"]]     #except value to be predicted  
y=dataset[["weight"]] 


model=lm.LinearRegression()
model.fit(x,y)  

print(model.predict([[160,20,80,10,9]])) #weight ozhich values ellam pass cheyyy