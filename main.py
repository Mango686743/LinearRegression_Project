import numpy as np
import pandas as pd
from linear_regression import LinearRegression

df = pd.read_csv("data.csv")

X_ = df[["study_hours"]].values
y_ = df["score"].values

model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X_,y_)

testcase = np.array([[15]])
res = model.predict(testcase)

print(res[0])