import joblib

model = joblib.load("fish_model.pkl")

y = [[242, 23.2, 25.4, 30, 11.52, 4.02]]
y_pred = model.predict(y)
species = round(y_pred[0][0])

print(species)
print("0=Bream, 1=Parkki,  2=Perch,  3=Pike,  4=Roach,  5=Smelt,  6=Whitefish")

