import joblib
import numpy as np

model = joblib.load("bank_model.pkl")



y = model.predict(np.array([[28,47,588,364]]))

y = int(np.round(y)[0][0])
print(y)

if y == 0:
    print("Loan Rejected")
else:
    print("Loan Approved")
