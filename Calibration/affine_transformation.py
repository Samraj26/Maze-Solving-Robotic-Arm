import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

# Points in Pixel coordinate
B = np.array([
    [475, 370, 1],
    [229, 118, 1],
    [516, 86, 1],
    [189, 403, 1],
    [349, 230, 1],
    [148, 315, 1],
    [104, 179, 1]
])

# Points in Robot coordinate
A = np.array([
    [-387, -335, 1],
    [-293, -230, 1],
    [-275, -351, 1],
    [-399, -208, 1],
    [-343, -291, 1],
    [-377, -195, 1],
    [-328, -167, 1]
])

A_coords = A[:, :2]
B_coords = B[:, :2]


B_matrix = np.hstack((B_coords, np.ones((B_coords.shape[0], 1))))

T, _, _, _ = lstsq(B_matrix, A_coords, rcond=None)

affine_transformation_matrix = np.vstack([T.T, [0, 0, 1]])
A_transformed = (affine_transformation_matrix@B.T).T

plt.figure(figsize=(10, 6))
plt.scatter(A[:, 0], A[:, 1], color='blue', label='A (Robot Points)')
plt.scatter(B[:, 0], B[:, 1], color='red', label='B (Pixel Points)')
plt.scatter(A_transformed[:, 0], A_transformed[:, 1], color='green', marker='x', label="B' (Transformed Points)")

# Draw lines connecting A and B'
for i in range(len(A)):
    plt.plot([A[i, 0], A_transformed[i, 0]], [A[i, 1], A_transformed[i, 1]], 'k--', linewidth=0.5)

# Set plot properties
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Original Points (A), Target Points (B), and Transformed Points (B')")
plt.axis("equal")
plt.grid(True)
plt.show()


np.save("transformation_matrix.npy",affine_transformation_matrix)
print(affine_transformation_matrix@np.array([320,210,1]).T)