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

a1 = int(input("Enter Age:"))
a2 = str(input("Enter Gender:"))
a3 = str(input("Enter Body type:"))
a4 = int(input("Enter Height:"))

m=le.fit_transform([a2])[0]
n=le.fit_transform([a3])[0]


print(model.predict([[a1,m,n,a4]]))