
# Project A16: Turbine exit temperature from other engine sensors

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# STEP 1: Load the data file
cols = ["unit", "cycle", "set1", "set2", "set3"] + ["s" + str(i) for i in range(1, 22)]

df = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None, names=cols)
print("Rows and columns:", df.shape)

# STEP 2: Clean the data
df = df.dropna(subset=["s3", "s7", "s9", "s11", "s12", "s4"])

# STEP 3: Choose inputs X and output y
inputs = ["s3", "s7", "s9", "s11", "s12"]
X = df[inputs]
y = df["s4"]

# STEP 4: Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# STEP 5: Train
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 6: Test
y_pred = model.predict(X_test)

print("R2 score     :", round(r2_score(y_test, y_pred), 3))
print("Average error:", round(mean_absolute_error(y_test, y_pred), 2), "deg R")

print("Effect of each input (slope):")
for name, m in zip(inputs, model.coef_):
    print(" ", name, "->", round(m, 4))

# STEP 7: Predict a new case and draw Actual vs Predicted
new = pd.DataFrame({
    "s3": [1590], "s7": [553.5], "s9": [9060],
    "s11": [47.5], "s12": [521.5]
})

print("Predicted T50:", round(model.predict(new)[0], 2), "deg R")

plt.scatter(y_test, y_pred, s=12, alpha=0.6)
low, high = y_test.min(), y_test.max()
plt.plot([low, high], [low, high], color="red", label="Perfect prediction")

plt.xlabel("Actual T50 (degrees R)")
plt.ylabel("Predicted T50 (degrees R)")
plt.title("Turbine exit temperature from other engine sensors")
plt.legend()
plt.show()
