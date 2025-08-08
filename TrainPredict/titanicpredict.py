import joblib
model = joblib.load("titanic_model.pkl")
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
lg = LabelEncoder()
lk = LabelEncoder()

print("coefficient:",model.coef_[0])
print("intercept:",model.intercept_[0])

m=le.fit_transform(["female"])[0]  
n=lg.fit_transform(["C0"])[0]
bk=lk.fit_transform(["S"])[0]

y= model.predict([[3,m,90,0,2,n,bk]])





if y>0.5 :
    print("This person survived")

else:
    print("This person did not survive")
