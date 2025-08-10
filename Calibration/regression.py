from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Input Data
B_coords = np.array([
    [475, 370],
    [229, 118],
    [516, 86],
    [189, 403],
    [349, 230],
    [148, 315],
    [104, 179]
])
A_coords = np.array([
    [-387, -335],
    [-293, -230],
    [-275, -351],
    [-399, -208],
    [-343, -291],
    [-377, -195],
    [-328, -167]
])

poly = PolynomialFeatures(degree=2, include_bias=False)
B_poly = poly.fit_transform(B_coords)

model = LinearRegression()
model.fit(B_poly, A_coords)

A_predicted = model.predict(B_poly)


plt.figure(figsize=(10, 6))
plt.scatter(A_coords[:, 0], A_coords[:, 1], color='blue', label='A (Robot Points)')
plt.scatter(B_coords[:, 0], B_coords[:, 1], color='red', label='B (Pixel Points)')
plt.scatter(A_predicted[:, 0], A_predicted[:, 1], color='green', marker='x', label="B' (Predicted Points)")

for i in range(len(A_coords)):
    plt.plot([A_coords[i, 0], A_predicted[i, 0]], [A_coords[i, 1], A_predicted[i, 1]], 'k--', linewidth=0.5)

plt.legend()
plt.xlabel("X (Robot Coordinates)")
plt.ylabel("Y (Robot Coordinates)")
plt.title("Mapping with Polynomial Regression")
plt.axis("equal")
plt.grid(True)
plt.show()


with open("polynomial_features.pkl", "wb") as f:
    pickle.dump(poly, f)
with open("polynomial_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully.")

