import numpy as np
import pandas as pd
from linear_regression import LinearRegression

df = pd.read_csv("data.csv")

X_test = df[["study_hours","gaming_hours","sleeping_hours"]].values     #List of dictionary -> Matrix of Weight
y_test = df["score"].values                                             #A vector of result y

model = LinearRegression(learning_rate=0.001,n_iters=1000)
model.fit(X_test,y_test)

A_crazy_student = np.array([[13,0.5,8]])           
res = model.predict(A_crazy_student)

#New bug: Maximum and Minimum of points is [0,100] -> Limits the res for pratical situation
res = np.clip(res,0,100)        #Not recommend

##Future Work Here

print(f"According to the reseacher, it's vividly that the student, who studies {A_crazy_student[0][0]} hours, plays game {A_crazy_student[0][1]} hours and sleeps {A_crazy_student[0][2]} hours, will gain {res[0]} out of 100")