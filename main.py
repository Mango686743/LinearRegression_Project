import numpy as np
import pandas as pd
from linear_regression import LinearRegression

#Đọc từ file csv bằng pandas
df = pd.read_csv("data.csv")

#Chuyển dữ liệu dataframe pandas -> arr numpy cho dễ tính toán
X = df[['study_hours']].values
y = df[['score']].values

model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X,y)

X_test = np.array([[15]])
print(model.predict(X_test)[0][0])
