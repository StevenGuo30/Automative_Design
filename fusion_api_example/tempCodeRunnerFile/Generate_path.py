import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import json
import os
import sys

from input_interface import input_interface

np.random.seed(42) # TODO: Remove this line for randomization

# TODO: Organize the code and combined it with the input interface
# TODO: Group 2 connections is a little bit strange, need to be fixed(at point 5, there is a sharp turn which should be smooth)



# ---------------- two methods to generate the spline ----------------
def generate_arc_3d(p0, p1, arc_height_ratio, num_points):
    p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    midpoint = (p0 + p1) / 2
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("p0 and p1 are the same point!")
    direction /= norm
    rand_vec = np.random.randn(3)
    rand_vec -= rand_vec.dot(direction) * direction
    rand_vec /= np.linalg.norm(rand_vec)
    perp=rand_vec
    offset_distance = arc_height_ratio * norm
    p_m = midpoint + perp * offset_distance
    a = p0 - p_m
    b = p1 - p_m
    cross_ab = np.cross(a, b)
    cross_norm = np.linalg.norm(cross_ab)    
    if cross_norm == 0:
        raise ValueError("p0, p1, and pm are colinear!")
    center = p_m + np.cross((np.linalg.norm(b)**2 * a - np.linalg.norm(a)**2 * b), cross_ab) / (2 * cross_norm**2)
    r = np.linalg.norm(p0 - center)
    u = (p0 - center) / r
    v = np.cross(cross_ab, u)
    v /= np.linalg.norm(v)
    theta0 = 0
    vec1 = (p1 - center) / r
    theta1 = np.arctan2(np.dot(vec1, v), np.dot(vec1, u))
    if theta1 < 0:
        theta1 += 2 * np.pi
    ts = np.linspace(theta0, theta1, num_points)
    arc = np.zeros((num_points, 3))
    for i, t in enumerate(ts):
        arc[i] = center + r * (np.cos(t) * u + np.sin(t) * v)
    end_tangent = arc[-1] - arc[-2]
    end_tangent /= np.linalg.norm(end_tangent)
    return arc, end_tangent

def generate_bezier_with_tangent(p0, p1, start_tangent, scale=0.3, num_points=50):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    start_tangent = np.array(start_tangent, dtype=float)
    start_tangent /= np.linalg.norm(start_tangent)

    chord = p1 - p0
    chord_length = np.linalg.norm(chord)
    p_m = p0 + scale * chord_length * start_tangent
    #  Bézier Curve
    ts = np.linspace(0, 1, num_points)
    bezier_curve = np.zeros((num_points, 3))   
    for idx, t in enumerate(ts):
        bezier_curve[idx] = (1 - t)**2 * p0 + 2 * (1 - t) * t * p_m + t**2 * p1
    end_tangent = bezier_curve[-1] - bezier_curve[-2]
    end_tangent /= np.linalg.norm(end_tangent)
    return bezier_curve, end_tangent




# ---------------- Interpolate Multiple Points with 3D Arcs ----------------
def interpolate_curve(points, arc_height_ratio, num_points):
    points = np.array(points)
    curve_points = []    
    if len(points) < 2:
        raise ValueError("Need at least 2 points to interpolate!")
    if len(points) == 2:
        arc, _ = generate_arc_3d(points[0], points[1], arc_height_ratio, num_points)
        curve_points.append(arc)
    else:
        arc, tangent = generate_arc_3d(points[0], points[1], arc_height_ratio, num_points)
        curve_points.append(arc)
        for i in range(1, len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            arc, tangent = generate_bezier_with_tangent(p0, p1, tangent, 0.33, num_points)
            curve_points.append(arc[1:])  
    full_curve = np.vstack(curve_points)
    return full_curve



# ---------------- Resample Curve ----------------
def resample_curve(curve, ds):
    lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    total_length = np.sum(lengths)
    n_samples = int(total_length / ds) + 1
    new_points = np.linspace(0, 1, n_samples)
    t = np.linspace(0, 1, len(curve))
    sampled_curve = np.vstack([
        np.interp(new_points, t, curve[:, 0]),
        np.interp(new_points, t, curve[:, 1]),
        np.interp(new_points, t, curve[:, 2])
    ]).T
    return sampled_curve

# ---------------- Collision Detection ----------------
def is_pipe_collision(curve1, curve2, pipe_radius, sample_ds=0.05):
    sampled1 = resample_curve(curve1, sample_ds)
    sampled2 = resample_curve(curve2, sample_ds)

    tree2 = cKDTree(sampled2)

    for p1 in sampled1:
        dist, _ = tree2.query(p1, k=1)
        if dist < 2 * pipe_radius:
            return True
    return False

# ---------------- Bounding Box (Optional) ----------------
def compute_bbox(curve):
    min_corner = np.min(curve, axis=0)
    max_corner = np.max(curve, axis=0)
    return (min_corner, max_corner)





# ------------------- Node  -------------------
class Node:
    def __init__(self, point, parent=None):
        self.point = np.array(point)
        self.parent = parent

# ------------------- RRT -------------------
def is_collision(p1, p2, obstacle_kdtree, safe_radius=0.2, step_size=0.05):
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        return False
    direction = direction / length
    steps = int(length / step_size)
    for i in range(steps + 1):
        p = p1 + i * step_size * direction
        dist, _ = obstacle_kdtree.query(p)
        if dist < safe_radius:
            return True
    return False


def rrt_path(start, goal, obstacle_points, 
              x_limits, y_limits, z_limits,
              max_iters=500, step_size=1, goal_sample_rate=0.1, safe_radius=0.2):

    obstacle_kdtree = cKDTree(obstacle_points)

    start_node = Node(start)
    goal_node = Node(goal)
    nodes = [start_node]

    for _ in range(max_iters):
        if np.random.rand() < goal_sample_rate:
            sample = goal
        else:
            sample = np.array([
                np.random.uniform(*x_limits),
                np.random.uniform(*y_limits),
                np.random.uniform(*z_limits)
            ])

        dists = [np.linalg.norm(n.point - sample) for n in nodes]
        nearest_node = nodes[np.argmin(dists)]

        direction = sample - nearest_node.point
        if np.linalg.norm(direction) == 0:
            continue
        direction = direction / np.linalg.norm(direction)
        new_point = nearest_node.point + step_size * direction

        if is_collision(nearest_node.point, new_point, obstacle_kdtree, safe_radius, step_size=step_size/2):
            continue

        new_node = Node(new_point, nearest_node)
        nodes.append(new_node)

        if np.linalg.norm(new_point - goal) < step_size:
            if not is_collision(new_point, goal, obstacle_kdtree, safe_radius, step_size=step_size/2):
                print(" RRT Found path!")
                goal_node.parent = new_node
                nodes.append(goal_node)

                path = []
                cur = goal_node
                while cur is not None:
                    path.append(cur.point)
                    cur = cur.parent
                path.reverse()
                return np.array(path)

    print(" RRT failed to find path!")
    return None

def smooth_path_with_bspline(path, num_points=100):
    path = np.array(path)
    if len(path) <= 3:
        # Not enough points for cubic spline, return original or linear interp
        return path.copy()

    tck, u = splprep([path[:,0], path[:,1], path[:,2]], s=0)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    smooth_path = np.vstack([x_fine, y_fine, z_fine]).T
    return smooth_path

def is_path_collision(curve, obstacle_kdtree, safe_radius=0.2, step_size=0.05):
    for i in range(len(curve) - 1):
        if is_collision(curve[i], curve[i+1], obstacle_kdtree, safe_radius, step_size):
            return True
    return False

def is_self_collision(curve, pipe_radius, sample_ds=0.05):
    sampled = resample_curve(curve, sample_ds)
    tree = cKDTree(sampled)

    for i, pt in enumerate(sampled):
        # search for points within 2 * pipe_radius
        idxs = tree.query_ball_point(pt, r=2 * pipe_radius)

        for j in idxs:
            if abs(i - j) > 2:  # exclude adjacent points
                return True
    return False


def global_smooth_and_check(curve_segments, obstacle_points, pipe_radius=0.2, num_points=300):
    if len(curve_segments) == 0:
        raise ValueError("No segments to smooth!")

    full_curve = np.vstack(curve_segments)
    smooth_curve = smooth_path_with_bspline(full_curve, num_points=num_points)
    obstacle_kdtree = cKDTree(obstacle_points)
    if is_path_collision(smooth_curve, obstacle_kdtree, safe_radius=pipe_radius):
        return smooth_curve, False
    else:
        return smooth_curve, True



# ---------------- Generate Pipe Paths ----------------
def generate_pipe_paths(point_dict, group_connections, pipe_radius, sample_ds, max_retries, max_insertions):
    base_points = {name: np.array(coord) for name, coord in point_dict.items()}
    curves = []
    frames = []
    all_success_curves = []

    for group_idx, group in enumerate(group_connections):
        print(f"\nProcessing group {group_idx+1}: {group}")
        group_points = [base_points[name] for name in group]

        success = False

        # --------------- Phase 1: Direct arc/interpolation with retry ---------------
        for attempt in range(max_retries):
            try:
                if len(group_points) == 2:
                    curve, _ = generate_arc_3d(group_points[0], group_points[1], arc_height_ratio, num_points)
                else:
                    curve = interpolate_curve(group_points, arc_height_ratio, num_points)

                collision = False

                # Check collision with existing curves
                for existing_curve in all_success_curves:
                    if is_pipe_collision(curve, existing_curve, pipe_radius, sample_ds):
                        collision = True
                        break

                # Check self-intersection
                if not collision and is_self_collision(curve, pipe_radius, sample_ds):
                    print(f"  Self-collision detected in group {group_idx+1} attempt {attempt+1}")
                    collision = True

                if not collision:
                    print(f"Group {group_idx+1} succeeded in {attempt+1} tries (no collision)!")
                    all_success_curves.append(curve)
                    frames.append(curve.copy())
                    success = True
                    break

            except Exception as e:
                print(f"Error in attempt {attempt+1}: {e}")

        # --------------- Phase 2: RRT-based path planning with smoothing ---------------
        if not success:
            print(f" Random seed retries failed for Group {group_idx+1}, trying segmented RRT + global smoothing...")

            if all_success_curves:
                obstacle_points = np.vstack(all_success_curves)
            else:
                obstacle_points = np.empty((0, 3))

            all_points = np.vstack(list(base_points.values()))
            margin = 0.5
            x_min, y_min, z_min = all_points.min(axis=0) - margin
            x_max, y_max, z_max = all_points.max(axis=0) + margin
            x_limits = (x_min, x_max)
            y_limits = (y_min, y_max)
            z_limits = (z_min, z_max)

            curve_segments = []

            for i in range(len(group_points) - 1):
                start = group_points[i]
                goal = group_points[i + 1]

                rrt_result = rrt_path(
                    start=start,
                    goal=goal,
                    obstacle_points=obstacle_points,
                    x_limits=x_limits,
                    y_limits=y_limits,
                    z_limits=z_limits,
                    max_iters=1500,
                    step_size=0.5,
                    goal_sample_rate=0.2,
                    safe_radius=pipe_radius
                )

                if rrt_result is None:
                    print(f" RRT failed between point {i} and {i+1}, giving up this group.")
                    break

                smoothed_segment = smooth_path_with_bspline(rrt_result, num_points=100)

                if is_path_collision(smoothed_segment, cKDTree(obstacle_points), safe_radius=pipe_radius):
                    print(f" Smoothed segment between {i} and {i+1} collides, giving up this group.")
                    break

                if i > 0:
                    smoothed_segment = smoothed_segment[1:]

                curve_segments.append(smoothed_segment)

            else:
                full_curve = np.vstack(curve_segments)
                globally_smoothed_curve = smooth_path_with_bspline(full_curve, num_points=300)

                if is_path_collision(globally_smoothed_curve, cKDTree(obstacle_points), safe_radius=pipe_radius):
                    print(f" Global smoothed curve collides with existing pipes, group {group_idx+1} failed.")
                elif is_self_collision(globally_smoothed_curve, pipe_radius, sample_ds):
                    print(f" Self-collision in smoothed curve, group {group_idx+1} failed.")
                else:
                    all_success_curves.append(globally_smoothed_curve)
                    frames.append(globally_smoothed_curve.copy())
                    print(f" Group {group_idx+1} succeeded with segmented RRT and global smoothing!")
                    success = True

        # --------------- Phase 3: Give up this group ---------------
        if not success:
            print(f" Group {group_idx+1} failed after all retries and insertions.")

    return frames, all_success_curves



# ---------------- Visualization ----------------
def visualize_pipe_animation(frames, points=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        
        for i in range(frame_idx + 1):
            curve_pts = np.array(frames[i])
            ax.plot(curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2], 'b-', linewidth=2)

        if points is not None:
            pts = np.array(points)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=50, marker='o', label='Input Points')
            for i, (x, y, z) in enumerate(pts):
                ax.text(x, y, z, f'{i}', fontsize=10, color='black')

        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        ax.set_zlim(-1, 2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Step {frame_idx+1}/{len(frames)}", fontsize=14)
        ax.grid(True)
        ax.legend()

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1500, repeat=False)
    plt.tight_layout()
    plt.show()

# ---------------- Save Spline into Json File ----------------
def save_splines_to_json(spline_list, output_path, num_sample_points=300):
    """
    Save spline paths to JSON after high-density resampling.

    Parameters:
    - spline_list: list of np.ndarray, each shape (N, 3)
    - output_path: str, where to save the JSON file
    - num_sample_points: int, number of points to sample per spline
    """
    serialized = []
    for spline in spline_list:
        t = np.linspace(0, 1, spline.shape[0])
        t_sample = np.linspace(0, 1, num_sample_points)

        resampled = np.vstack([
            np.interp(t_sample, t, spline[:, 0]),
            np.interp(t_sample, t, spline[:, 1]),
            np.interp(t_sample, t, spline[:, 2])
        ]).T

        serialized.append(resampled.tolist())

    with open(output_path, 'w') as f:
        json.dump(serialized, f, indent=2)




# ---------------- Complex Test Case ----------------
def test_complex_case(pipe_radius=0.01, sample_ds=0.05, max_retries=10, max_insertions=5):
    point_dict = {
        'A': [0, 0, 0],
        'B': [0, 1, 0],
        'C': [1, 1, 0],
        'D': [1, 0, 0],
        'E': [0.5, 1, 0.2],
        'F': [0.5, 0.5, 0.5],
        'G': [0.25, 0.75, 0.8],
        'H': [1.2, 1.2, 0.1],
        'I': [1.2, 0.2, 0.1],
        'J': [-0.3, 0.8, 0.2],
        'K': [0.5, 0.5, -0.5],
    }
    group_connections = [
        ['A', 'C', 'E', 'F', 'G'],
        ['B', 'D', 'H', 'I'],
        ['J', 'K']
    ]
    
    frames, curves = generate_pipe_paths(
        point_dict, 
        group_connections, 
        pipe_radius,      
        sample_ds,        
        max_retries,        
        max_insertions      
    )
    visualize_pipe_animation(frames)


# ---------------- Main Function ----------------
if __name__ == "__main__":
    # read point_dict and group_connections from paired_points.json
    # if not exist, use input_interface to generate it
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    json_path = os.path.join(script_dir, "paired_points.json")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        point_dict = {}          # point name and coordinates
        group_connections = []   # group name and point names

        for group in data:
            group_names = []
            for point in group:
                name = point["name"]
                coords = point["coordinates"]
                point_dict[name] = [coords["x"], coords["y"], coords["z"]]
                group_names.append(name)
            group_connections.append(group_names)
        print(f"Loaded {len(group_connections)} groups from {json_path}.")
    else:
        print(f"{json_path} not found, generating points with input interface...")
        point_dict, group_connections = input_interface()
        with open(json_path, "w") as f:
            json.dump(group_connections, f, indent=2)
        print(f"Saved {len(group_connections)} groups to {json_path}.")
    
    arc_height_ratio=0.22
    num_points=50
    pipe_radius=0.01
    sample_ds=0.05
    max_retries=10  #30
    max_insertions=5


    frames, curves = generate_pipe_paths(
        point_dict, 
        group_connections, 
        pipe_radius,      
        sample_ds,        
        max_retries,        
        max_insertions      
    )
    
    print(f"Generated {len(curves)} curves.")
    # Save the spline paths to a JSON file
    save_path = os.path.join(os.path.dirname(__file__), "exported_splines.json")
    save_splines_to_json(curves, save_path, num_sample_points=300)
    print(f"Spline paths saved to: {save_path}")

    point_names = sum(group_connections, [])  
    points = [point_dict[name] for name in point_names]

    visualize_pipe_animation(frames, points=points)
