import joblib
import numpy as np

model = joblib.load("student_model.pkl")



y = model.predict(np.array([[15,4,3,2]]))

y = int(np.round(y)[0][0])
print(y)

print("Grade:")
if y == 0:
    print("A")
elif y == 1:
    print("B")
else:
    print("C")
