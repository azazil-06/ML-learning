import joblib
import numpy as np

model = joblib.load("realestate_model.pkl")



y = model.predict(np.array([[3285,1,2,2,0]]))

print(y)