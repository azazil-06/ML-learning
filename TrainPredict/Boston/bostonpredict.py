import joblib

model = joblib.load("boston_model.pkl")






y = model.predict([[0.02731, 0, 7.07, 0, 0.469, 6.421, 78.9, 4.9671, 2, 242, 17.8, 396.9, 9.14]])


print("Predicted price (medv):", y)
