import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# Define custom colormap
colors = [(1, 1, 1), (1, 0, 0)]  # White to Red
custom_cmap = LinearSegmentedColormap.from_list("white_red", colors)
colors2 = [(1, 1, 1), (1, 1, 1)]  # White
custom_cmap2 = LinearSegmentedColormap.from_list("white", colors2)
# Rectangle class for bounding boxes
class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x  # Center x
        self.y = y  # Center y
        self.w = w  # Width
        self.h = h  # Height

    def contains(self, point):
        px, py = point
        return (self.x - self.w/2 <= px <= self.x + self.w/2 and
                self.y - self.h/2 <= py <= self.y + self.h/2)

# Quadtree implementation
class Quadtree:
    def __init__(self, boundary, max_points=5, min_size=1, max_cell_size=None, depth=0):
        self.boundary = boundary
        self.max_points = max_points
        self.min_size = min_size
        self.max_cell_size = max_cell_size
        self.points = []
        self.values = []
        self.divided = False
        self.depth = depth
        self.northeast = self.northwest = self.southeast = self.southwest = None

    def insert(self, point, value):
        if not self.boundary.contains(point):
            return False
        
        should_subdivide = (
            len(self.points) >= self.max_points or
            (self.max_cell_size and
            (self.boundary.w > self.max_cell_size or self.boundary.h > self.max_cell_size))
        )

        if not should_subdivide or self.boundary.w <= self.min_size:
            self.points.append(point)
            self.values.append(value)
            return True
        
        if not self.divided:
            self.subdivide()
        
        for child in [self.northeast, self.northwest, self.southeast, self.southwest]:
            if child.insert(point, value):
                return True
        return False


    def subdivide(self):
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h
        new_w, new_h = w / 2, h / 2
        
        self.northeast = Quadtree(Rectangle(x + new_w/2, y - new_h/2, new_w, new_h), self.max_points, self.min_size, self.depth + 1)
        self.northwest = Quadtree(Rectangle(x - new_w/2, y - new_h/2, new_w, new_h), self.max_points, self.min_size, self.depth + 1)
        self.southeast = Quadtree(Rectangle(x + new_w/2, y + new_h/2, new_w, new_h), self.max_points, self.min_size, self.depth + 1)
        self.southwest = Quadtree(Rectangle(x - new_w/2, y + new_h/2, new_w, new_h), self.max_points, self.min_size, self.depth + 1)
        self.divided = True

    def overlaps(self, rect1, rect2):
        return not (rect1.x + rect1.w/2 < rect2.x - rect2.w/2 or
                    rect1.x - rect1.w/2 > rect2.x + rect2.w/2 or
                    rect1.y + rect1.h/2 < rect2.y - rect2.h/2 or
                    rect1.y - rect1.h/2 > rect2.y + rect2.h/2)

    
    def get_cells(self, return_points=False):
        if not self.divided:
            if self.points:
                if len(self.points) == 1 and np.mean(self.values) > 1: #probably just an error
                    return []
                avg_val = np.mean(self.values)
                result = (self.boundary, avg_val)
                if return_points:
                    return [(self.boundary, avg_val, self.points, self.values)]
                return [result]
            return []
        else:
            cells = []
            for child in [self.northeast, self.northwest, self.southeast, self.southwest]:
                cells.extend(child.get_cells(return_points))
            return cells


# Load and process data
points1 = np.loadtxt('recon1.txt') # C:/Users/panag/Desktop/labwork/change_detection/
points2 = np.loadtxt('recon2.txt')
all_points = np.vstack([points1, points2])

# Compute nearest-neighbor distances
kdtree1 = KDTree(points1)
kdtree2 = KDTree(points2)
d1, _ = kdtree2.query(points1, k=1)
d2, _ = kdtree1.query(points2, k=1)
all_distances = np.concatenate([d1, d2])

# Define quadtree boundary
x_min, y_min = np.min(all_points, axis=0)
x_max, y_max = np.max(all_points, axis=0)
boundary = Rectangle((x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min)
max_size = (x_max - x_min) / 10  # for example, divide space into at least 10x10 grid
min_size =(x_max - x_min)/50
quadtree1 = Quadtree(boundary, max_points=5, min_size=min_size, max_cell_size=max_size)
quadtree2 = Quadtree(boundary, max_points=5, min_size=min_size, max_cell_size=max_size)



# Insert points into respective quadtrees
for point, dist in zip(points1, d1):
    quadtree1.insert(point, dist)
for point, dist in zip(points2, d2):
    quadtree2.insert(point, dist)

threshold = 0.2  # Set your mismatch threshold here

quadtree3 = Quadtree(boundary, max_points=5, min_size=min_size, max_cell_size=max_size) # this will be the day1 data updated with the day 2 data when the day2 differ from day 1
points3 = []

quadtree4 = Quadtree(boundary, max_points=5, min_size=min_size, max_cell_size=max_size) # this will be the intersection of day 1 and day 2
points4 = []
# Get cells with point data
cells1 = quadtree1.get_cells(return_points=True)
cells2 = quadtree2.get_cells(return_points=True)

# Build lookup dictionary from tree2 for fast match
def cell_key(bounds):
    return (round(bounds.x, 6), round(bounds.y, 6), round(bounds.w, 6), round(bounds.h, 6))

tree2_dict = {cell_key(b): (avg, pts, vals) for b, avg, pts, vals in cells2}
tree1_dict = {cell_key(b): (avg, pts, vals) for b, avg, pts, vals in cells1}
# Decide which data to insert into quadtree3
for b1, avg1, pts1, vals1 in cells1:
    key = cell_key(b1)
    if key in tree2_dict:
        avg2, pts2, vals2 = tree2_dict[key]
        if avg1 <= threshold and avg2 <= threshold:
            for p, v in zip(pts1, vals1):
                quadtree3.insert(p, v)
                points3.append(p)
                quadtree4.insert(p, v)
                points4.append(p)
        else:
            for p, v in zip(pts2, vals2):
                quadtree3.insert(p, v)
                points3.append(p)
                quadtree4.insert(p, v)
                points4.append(p)
# now the quadtree3 is the Intersection of 1 and 2 (with a threshold)
# to update the tree i will add the points that are in 2 but not in 1.
for b2, avg2, pts2, vals2 in cells2:
    key = cell_key(b2)
    if key not in tree1_dict:
        if avg1 >= threshold:
            for p, v in zip(pts2, vals2):
                quadtree3.insert(p, v)
                points3.append(p)
        
            

             
# Convert list of points to NumPy array
points3 = np.array(points3)
points4 = np.array(points4)


# Visualization
fig, axs = plt.subplots(1, 2, figsize=(18, 9))
titles = ["Quadtree for Set 1 (d1)", "Quadtree for Set 2 (d2)"]
quadtrees = [quadtree1, quadtree2]
point_sets = [points1, points2]


for ax, qt, title, pts in zip(axs, quadtrees, titles, point_sets):
    cells = qt.get_cells()
    values = [val for (_, val) in cells]
    norm = Normalize(vmin=0, vmax=np.percentile(values, 95))  # Normalize per subplot
    
    for bounds, avg_dist in cells:
        rect = MplRectangle((bounds.x - bounds.w/2, bounds.y - bounds.h/2), bounds.w, bounds.h,
                            facecolor=custom_cmap(norm(avg_dist)), alpha=0.8, edgecolor='none')
        ax.add_patch(rect)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Mismatch Score')
    
    ax.scatter(pts[:, 0], pts[:, 1], s=3, c='grey', alpha=0.7)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.set_title(title)

#quadtree1 = quadtree3

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 2, figsize=(18, 9))
titles = ["updated Quadtree1", "Intersection of day 1 and day 2"]
quadtrees = [quadtree3, quadtree4]
point_sets = [points3, points4]

for ax, qt, title, pts in zip(axs, quadtrees, titles, point_sets):
    cells = qt.get_cells()
    values = [val for (_, val) in cells]
    norm = Normalize(vmin=0, vmax=np.percentile(values, 95))  # Normalize per subplot
    
    for bounds, avg_dist in cells:
        rect = MplRectangle((bounds.x - bounds.w/2, bounds.y - bounds.h/2), bounds.w, bounds.h,
                            facecolor=custom_cmap2(norm(avg_dist)), alpha=0.8, edgecolor='none')
        ax.add_patch(rect)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap2)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Mismatch Score')
    
    ax.scatter(pts[:, 0], pts[:, 1], s=3, c='grey', alpha=0.7)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.set_title(title)

plt.tight_layout()
plt.show()

# Cluster τα points3 (updated quadtree points)
eps = 0.15  # μέγιστη απόσταση για να θεωρούνται σημεία στον ίδιο πυρήνα
min_samples = 3  # ελάχιστος αριθμός σημείων για να θεωρηθεί "πυκνή" περιοχή (να θεωρηθεί αντικείμενο)
db = DBSCAN(eps=eps, min_samples=min_samples).fit(points3)
labels = db.labels_  # Κάθε σημείο έχει μια ετικέτα (-1 για noise)

# Αριθμός clusters (αγνοώντας το -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Detected clusters: {n_clusters}")

# Assign colors
unique_labels = set(labels)
colors = plt.colormaps.get_cmap('tab20').resampled(len(unique_labels))  # διαφορετικό χρώμα για κάθε cluster

# Plotting
plt.figure(figsize=(10, 8))
for k in unique_labels:
    class_member_mask = (labels == k)
    xy = points3[class_member_mask]
    if k == -1:
        color = 'k'  # black for noise
        label = "Noise"
    else:
        color = colors(k)
        label = f"Cluster {k}"
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color,
             markeredgecolor=color, markersize=3, label=label)

plt.title(f"DBSCAN Clustering (ε={eps}, min_samples={min_samples})\n on the updated quadtree 1")
plt.xlabel("X")
plt.ylabel("Y")
#plt.legend(loc='best', markerscale=2, fontsize='small')
plt.gca().set_aspect('equal')
plt.grid(True)
plt.tight_layout()
plt.show()