import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev

Debug_Flag = False

class Node:
    def __init__(self, point, parent=None):
        self.point = np.array(point)
        self.parent = parent
        self.cost = 0.0 if parent is None else parent.cost + np.linalg.norm(self.point - parent.point)

def aabb_from_line(p1, p2, radius):
    center = (p1 + p2) / 2
    half_size = np.abs(p2 - p1) / 2 + radius
    return center, half_size

def lightweight_swept_overlap(p1, p2, obb_list, pipe_radius, entry_points=None, tolerance=1):
    """
    Fast swept volume vs OBB AABB overlap check.
    Allows exceptions for entry_points near OBB centers.
    """
    if not obb_list:
        return False

    swept_center, swept_half = aabb_from_line(p1, p2, pipe_radius)

    for obb in obb_list:
        obb_center = obb["center"]
        obb_half = (np.max(obb["corners"], axis=0) - np.min(obb["corners"], axis=0)) / 2

        # ✅ Allow entry contact
        if entry_points is not None:
            for p in entry_points:
                dist = np.linalg.norm(p - obb_center)
                if dist < tolerance:
                    if Debug_Flag:
                        print(f"[RRT] Entry contact accepted: point {p} is within {dist:.4f} of center of OBB {obb['name']}")
                    break  # skip this OBB
            else:
                delta = np.abs(swept_center - obb_center)
                if np.all(delta <= (swept_half + obb_half)):
                    if Debug_Flag:
                        print(f"[RTT Reject] Swept-overlap with OBB {obb['name']} between {p1} → {p2}")
                    return True

    return False



def is_collision(p1, p2, obstacle_kdtree, safe_radius=0.2, step_size=0.05, obb_list=None, pipe_radius=None):
    """
    Check collision between p1 and p2 using KDTree + optional AABB-based swept check.
    Returns True if collision detected.
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        print(f"[Reject] Zero-length edge: {p1}")
        return True
    direction = direction / length
    steps = int(length / step_size)

    for i in range(steps + 1):
        p = p1 + i * step_size * direction
        dist, _ = obstacle_kdtree.query(p)
        if dist < safe_radius:
            if Debug_Flag:
                print(f"[RTT Reject] KDTree collision at step {i}: distance {dist:.4f} < radius {safe_radius:.4f}")
            return True

    if obb_list and pipe_radius is not None:
       if lightweight_swept_overlap(p1, p2, obb_list, pipe_radius, entry_points=[p1, p2]):
            if Debug_Flag:
                print(f"[RTT Reject] Swept-overlap with OBB between {p1} → {p2}")
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
    tck, u = splprep([path[:, 0], path[:, 1], path[:, 2]], s=0)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    return np.vstack([x_fine, y_fine, z_fine]).T


def obb_list_to_pointcloud(obb_list, samples_per_face=4):
    """
    Convert OBB boxes to surface point clouds.
    """
    if not obb_list:
        return np.empty((0, 3))

    all_points = []
    for obb in obb_list:
        corners = obb["corners"]
        faces = [
            [corners[i] for i in [0, 1, 3, 2]],
            [corners[i] for i in [4, 5, 7, 6]],
            [corners[i] for i in [0, 1, 5, 4]],
            [corners[i] for i in [2, 3, 7, 6]],
            [corners[i] for i in [1, 3, 7, 5]],
            [corners[i] for i in [0, 2, 6, 4]],
        ]
        for face in faces:
            f = np.array(face)
            for u in np.linspace(0, 1, samples_per_face):
                for v in np.linspace(0, 1, samples_per_face):
                    p = (1-u)*(1-v)*f[0] + u*(1-v)*f[1] + u*v*f[2] + (1-u)*v*f[3]
                    all_points.append(p)
    return np.array(all_points)

def rrt_path(start, goal, obstacle_points, x_limits, y_limits, z_limits,
             max_iters=1000, step_size=0.5, goal_sample_rate=0.1,
             safe_radius=0.2, pipe_radius = 0.1,obb_list=None):
    """
    RRT* path planning with obstacle point cloud and optional OBB obstacles.
    """
    obstacle_kdtree = cKDTree(obstacle_points)
    nodes = [Node(start)]
    goal_node = Node(goal)

    print(f"\n[RRT] Start: {start}, Goal: {goal}, Step size: {step_size}, Radius: {pipe_radius}, Safe_radius: {safe_radius}")
    print(f"[RRT] X limit: {x_limits}, Y limit: {y_limits}, Z limit: {z_limits}")

    radius = step_size * 2
    success_samples = 0
    rejection_count = 0
    connection_attempts = 0

    for iteration in range(max_iters):
        sample = goal if np.random.rand() < goal_sample_rate else np.array([
            np.random.uniform(*x_limits),
            np.random.uniform(*y_limits),
            np.random.uniform(*z_limits)
        ])

        nearest_node = min(nodes, key=lambda n: np.linalg.norm(n.point - sample))
        direction = sample - nearest_node.point
        if np.linalg.norm(direction) == 0:
            rejection_count += 1
            continue
        new_point = nearest_node.point + step_size * direction / np.linalg.norm(direction)

        if is_collision(nearest_node.point, new_point, obstacle_kdtree,
                        safe_radius, step_size / 2,
                        obb_list=obb_list, pipe_radius=pipe_radius):
            rejection_count += 1
            continue

        new_node = Node(new_point)
        near_nodes = [n for n in nodes if
                      np.linalg.norm(n.point - new_point) < radius and
                      not is_collision(n.point, new_point, obstacle_kdtree,
                                       safe_radius, step_size / 2,
                                       obb_list=obb_list, pipe_radius=safe_radius)]

        if near_nodes:
            best_parent = min(near_nodes, key=lambda n: n.cost + np.linalg.norm(n.point - new_point))
            new_node.parent = best_parent
            new_node.cost = best_parent.cost + np.linalg.norm(best_parent.point - new_point)

        nodes.append(new_node)
        success_samples += 1

        # Rewiring
        for n in near_nodes:
            new_cost = new_node.cost + np.linalg.norm(new_node.point - n.point)
            if new_cost < n.cost and not is_collision(n.point, new_node.point, obstacle_kdtree,
                                                       safe_radius, step_size / 2,
                                                       obb_list=obb_list, pipe_radius=safe_radius):
                n.parent = new_node
                n.cost = new_cost

        # Try to connect to goal
        if np.linalg.norm(new_node.point - goal) < step_size:
            connection_attempts += 1
            if not is_collision(new_node.point, goal, obstacle_kdtree,
                                safe_radius, step_size / 2,
                                obb_list=obb_list, pipe_radius=safe_radius):
                goal_node.parent = new_node
                path = []
                cur = goal_node
                while cur:
                    path.append(cur.point)
                    cur = cur.parent
                print(f"[RRT] Goal successfully connected after {iteration+1} iterations")
                print(f"[RRT] Sampled nodes: {len(nodes)}, Accepted: {success_samples}, Rejected: {rejection_count}")
                return np.array(path[::-1])
            else:
                print(f"[RRT] Tried to connect to goal but failed due to collision.")

    print(f"[RRT] Failed to connect goal after {max_iters} iterations.")
    print(f"[RRT] Sampled nodes: {len(nodes)}, Accepted: {success_samples}, Rejected: {rejection_count}, Goal attempts: {connection_attempts}")
    return None
