import joblib
import numpy as np

model = joblib.load("IOT_model.pkl")



y = model.predict(np.array([[2000.2,1.01,90.32,6.97]]))

x = int(np.round(y)[0][0])
print("Faulty value",x)
if x == 1:
    print("Device is faulty")
else:
    print("Device is not faulty")