import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Load data
points1 = np.loadtxt('recon1.txt')
points2 = np.loadtxt('recon2.txt')

# Define colors
col = np.array([0.812785127631716, 0.567494443643431, 0.933046653204004])
col1 = col
col2 = 1 - col

# Distance threshold
thres = 0.3

# KDTree for nearest-neighbor distances
kdtree1 = KDTree(points1)
kdtree2 = KDTree(points2)

# Compute distances
d1, _ = kdtree2.query(points1)
d2, _ = kdtree1.query(points2)

# Identify mismatches
b1 = d1 > thres
b2 = d2 > thres

# Assign colors (red for mismatches in set1, green for set2)
cols1 = np.where(b1[:, np.newaxis], [1, 0, 0], 0.5 * col1)
cols2 = np.where(b2[:, np.newaxis], [0, 1, 0], 0.5 * col2)

# Plot original points
plt.figure(figsize=(8, 8))
plt.scatter(points1[:, 0], points1[:, 1], s=3, c=[col1], marker='o')
plt.scatter(points2[:, 0], points2[:, 1], s=3, c=[col2], marker='o')
plt.gca().set_aspect('equal')
plt.title("Figure 1. Original Reconstructions of the Two Point Sets")
plt.show()

# Plot with mismatch highlights
plt.figure(figsize=(8, 8))
plt.scatter(points1[:, 0], points1[:, 1], s=2, c=cols1, marker='o')
plt.scatter(points2[:, 0], points2[:, 1], s=2, c=cols2, marker='o')
plt.gca().set_aspect('equal')
plt.axis('off')
plt.title("Figure 2. Point-Level Differences Between Reconstructions (Red = Missing, Green = New)")
plt.show()
