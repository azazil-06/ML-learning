import pandas as pd

import sklearn.neighbors as knn
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder


mydata=pd.read_csv("titanic.csv")

le = LabelEncoder()
lg = LabelEncoder()
lk = LabelEncoder()

mydata["Cabin_encoded"] = le.fit_transform(mydata[["Cabin"]])
mydata["Embarked_encoded"] = lg.fit_transform(mydata[["Embarked"]])
mydata["Sex_encoded"] = lk.fit_transform(mydata[["Sex"]])

# Drop rows with any NaN values
mydata = mydata.dropna()

x = mydata[["Pclass","Sex_encoded","Age","SibSp","Parch","Cabin_encoded","Embarked_encoded"]]
y = mydata[["Survived"]]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = knn.KNeighborsClassifier(n_neighbors=2)
model.fit(x_train,y_train)


y_pred = model.predict(x_test)
print("Accuracy sccore =", accuracy_score(y_test,y_pred))


m=le.fit_transform(["male"])[0]  #array 1st/sngl string index pass cheyyan use this
n=lg.fit_transform(["C24"])[0]
bk=lk.fit_transform(["S"])[0]

y= model.predict([[3,m,22,1,0,n,bk]])
print (y)





if y == 1 :
    print("This person survived")

else:
    print("This person did not survive")