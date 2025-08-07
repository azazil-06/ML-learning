import pandas as pd
import sklearn.linear_model as lm
from sklearn.preprocessing import LabelEncoder




dataset = pd.read_csv("persons.csv")


le = LabelEncoder()
dataset["gender_encoded"] = le.fit_transform(dataset[["Gender"]]) 
dataset["bodytype_encoded"] = le.fit_transform(dataset[["BodyType"]])



#----------------------------

x=dataset[["Age","gender_encoded","bodytype_encoded","Height"]]       
y=dataset["Weight"] 



model=lm.LinearRegression()
model.fit(x,y) 

m=le.fit_transform(["Male"])[0]
n=le.fit_transform(["Slim"])[0]


print(model.predict([[25,m,n,170]]))