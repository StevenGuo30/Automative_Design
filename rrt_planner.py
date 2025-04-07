
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev

class Node:
    def __init__(self, point, parent=None):
        self.point = np.array(point)
        self.parent = parent

def is_collision(p1, p2, obstacle_kdtree, safe_radius=0.2, step_size=0.05):
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0: return False
    direction = direction / length
    steps = int(length / step_size)
    for i in range(steps + 1):
        p = p1 + i * step_size * direction
        dist, _ = obstacle_kdtree.query(p)
        if dist < safe_radius:
            return True
    return False

def is_path_collision(curve, obstacle_kdtree, safe_radius=0.2, step_size=0.05):
    for i in range(len(curve) - 1):
        if is_collision(curve[i], curve[i+1], obstacle_kdtree, safe_radius, step_size):
            return True
    return False

def smooth_path_with_bspline(path, num_points=100):
    path = np.array(path)
    if len(path) <= 3:
        return path.copy()
    tck, u = splprep([path[:,0], path[:,1], path[:,2]], s=0)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    return np.vstack([x_fine, y_fine, z_fine]).T

def rrt_path(start, goal, obstacle_points,
             x_limits, y_limits, z_limits,
             max_iters=500, step_size=1.0, goal_sample_rate=0.1, safe_radius=0.2):
    obstacle_kdtree = cKDTree(obstacle_points)
    start_node = Node(start)
    goal_node = Node(goal)
    nodes = [start_node]

    for _ in range(max_iters):
        sample = goal if np.random.rand() < goal_sample_rate else np.array([
            np.random.uniform(*x_limits),
            np.random.uniform(*y_limits),
            np.random.uniform(*z_limits)
        ])
        nearest_node = min(nodes, key=lambda n: np.linalg.norm(n.point - sample))
        direction = sample - nearest_node.point
        if np.linalg.norm(direction) == 0: continue
        new_point = nearest_node.point + step_size * direction / np.linalg.norm(direction)

        if is_collision(nearest_node.point, new_point, obstacle_kdtree, safe_radius, step_size=step_size/2):
            continue

        new_node = Node(new_point, nearest_node)
        nodes.append(new_node)

        if np.linalg.norm(new_point - goal) < step_size and not is_collision(new_point, goal, obstacle_kdtree, safe_radius):
            goal_node.parent = new_node
            path = []
            cur = goal_node
            while cur:
                path.append(cur.point)
                cur = cur.parent
            return np.array(path[::-1])

    return None
