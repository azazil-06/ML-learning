#pip install -r \MLprograms\requirements.txt
#tensorflow


#--IMP---
#!  weight.py in Weight-predictor uses encoding in input


#-----------------------------:)----------------------------------#
"""                              DAY 01                                                      """
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

#----- !!!
#Encoding string data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
mydata["education_qualification_encoded"] = le.fit_transform(mydata[["education_qualification"]]) #x inte string value encoded into number

x = mydata[["education_qualification_encoded"]]
y = mydata[["salary"]]

#   oru single variable encode akkan DO LIKE THIS
m=le.fit_transform(["Male"])[0]  #smtg passing 0th index of this array only



#-------------------------------------------------------------------------------------------------------
"""                                     DAY 02 :)                                                           """

# Neighbors regressor (knn)

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

print(model.predict([[160]]))            # y ann ivide prdct akunne


#--------------------------------------------------------------------------------------------------
#multiple values varuannel How?

dataset = pd.read_csv("weight_predict_real.csv")

x=dataset[["height","age","bmi","muscle_mass","body_fat"]]     #except value to be predicted  
y=dataset[["weight"]] # value to be predicted


model=lm.LinearRegression()
model.fit(x,y)  

print(model.predict([[160,20,80,10,9]])) #weight ozhich values ellam pass cheyyy , nt:- as same in x datast above


#----------------------------------------------------------------

#to read & pass values ,,  input edukkan


g = int(input("Enter height:"))
getAge = int(input("Enter age:"))
getBMI = int(input("Enter BMI:"))
getMuscleMass = int(input("Enter musclemass:"))
getbodyfat = int(input("Enter bodyfat %:")) 

print(model.predict([[getHeight,getAge,getBMI,getMuscleMass,getbodyfat]]))

#----------------------------------------------------------------------------------------
"""                                  DAY 03                                                    """

"""
Accuracy edukan dataset split aknm

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

above line ath cheyyum in 80:20 rto

model.fit(x_train,y_train)
pass cheyunne splitted values only

"""
#Accuracy check KNN

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import sklearn.neighbors as knn


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = knn.KNeighborsClassifier(n_neighbors=2)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy sccore =", accuracy_score(y_test,y_pred))

#------------------------------------------------
#Accuracy check linear


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import math 


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)


pt.scatter(x_train,y_train)
pt.show()
model = lm.LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("mse= ", mean_squared_error(y_test,y_pred))
print("rmse= ",math.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 score= ",r2_score(y_test,y_pred))







































#model creation (pickle file)

import joblib
joblib.dump(model,"titanic_model.pkl")










#                     :)hi