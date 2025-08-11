import joblib
import numpy as np

model = joblib.load("wheather_model.pkl")



y = model.predict(np.array([[23.8,50,1.5,1030.7,1]]))

y = int(np.round(y)[0][0])


if y == 0:
    print("It will rain today")
else:
    print("It will not rain today")