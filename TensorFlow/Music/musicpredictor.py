import joblib
import numpy as np

model = joblib.load("music_model.pkl")



y = model.predict(np.array([[40,0,1,0.9]]))

y = int(np.round(y)[0][0])
print(y)

if y == 0:
    print("Chill")
elif y == 1:
    print("Focus")
elif y == 2:
    print("Party")
else:
    print("Workout")