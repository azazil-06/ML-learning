import pandas as pd
import sklearn.linear_model as lm
from sklearn.preprocessing import LabelEncoder




dataset = pd.read_csv("cancer.csv")


le = LabelEncoder()
dataset["dignosis_encoded"] = le.fit_transform(dataset[["diagnosis"]]) 




#----------------------------

g1 = float(input("Enter Radius mean: "))
g2 = float(input("Enter Texture mean: "))
g3 = float(input("Enter Perimeter mean: "))
g4 = float(input("Enter Area mean: "))
g5 = float(input("Enter Smoothness mean: "))
g6 = float(input("Enter Compactness mean: "))
g7 = float(input("Enter Concavity mean: "))
g8 = float(input("Enter Concave points mean: "))
g9 = float(input("Enter Symmetry mean: "))
g10 = float(input("Enter Fractal dimension mean: "))
g11 = float(input("Enter Radius SE: "))
g12 = float(input("Enter Texture SE: "))
g13 = float(input("Enter Perimeter SE: "))
g14 = float(input("Enter Area SE: "))
g15 = float(input("Enter Smoothness SE: "))
g16 = float(input("Enter Compactness SE: "))
g17 = float(input("Enter Concavity SE: "))
g18 = float(input("Enter Concave points SE: "))
g19 = float(input("Enter Symmetry SE: "))
g20 = float(input("Enter Fractal dimension SE: "))
g21 = float(input("Enter Radius worst: "))
g22 = float(input("Enter Texture worst: "))
g23 = float(input("Enter Perimeter worst: "))
g24 = float(input("Enter Area worst: "))
g25 = float(input("Enter Smoothness worst: "))
g26 = float(input("Enter Compactness worst: "))
g27 = float(input("Enter Concavity worst: "))
g28 = float(input("Enter Concave points worst: "))
g29 = float(input("Enter Symmetry worst: "))
g30 = float(input("Enter Fractal dimension worst: "))



x=dataset[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"]]   
   
y=dataset["dignosis_encoded"] 

#-----------------------


model=lm.LinearRegression()
model.fit(x,y)  

#-------------------------------------------



print(model.predict([[g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g21,g22,g23,g24,g25,g26,g27,g28,g29,g30]]))

if (y > 1).any():
    print("This person has cancer")

else:
     print("This person does not have cancer")