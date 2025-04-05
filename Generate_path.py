import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

# two methods to generate the spline
def generate_arc_3d(p0, p1, arc_height_ratio=0.22, num_points=50):
    p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    midpoint = (p0 + p1) / 2
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("p0 and p1 are the same point!")
    direction /= norm
    np.random.seed(42)
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
def interpolate_curve(points, arc_height_ratio=0.22, num_points=50):
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

# ---------------- Generate Pipe Paths ----------------
def generate_pipe_paths(point_dict, group_connections, pipe_radius=0.15, sample_ds=0.05, max_retries=10):
    base_points = {name: np.array(coord) for name, coord in point_dict.items()}
    curves = []
    frames = []
    all_success_curves = []
    
    for group_idx, group in enumerate(group_connections):
        print(f"\nProcessing group {group_idx+1}: {group}")
        group_points = [base_points[name] for name in group]

        success = False
        for attempt in range(max_retries):
            np.random.seed(attempt)  

            try:
                if len(group_points) == 2:
                    curve, _ = generate_arc_3d(group_points[0], group_points[1])
                else:
                    curve = interpolate_curve(group_points)

                collision = False
                for existing_curve in all_success_curves:
                    if is_pipe_collision(curve, existing_curve, pipe_radius, sample_ds):
                        collision = True
                        break

                if not collision:
                    print(f"Group {group_idx+1} succeeded in {attempt+1} tries!")
                    all_success_curves.append(curve)
                    frames.append(curve.copy())
                    success = True
                    break

            except Exception as e:
                print(f"Error in attempt {attempt+1}: {e}")

        if not success:
            print(f"Group {group_idx+1} failed to find collision-free path after {max_retries} attempts.")

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




# ---------------- Complex Test Case ----------------
def test_complex_case():
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
    frames, adjusted_points, edges = generate_pipe_paths(point_dict, group_connections, pipe_d=0.15)
    visualize_pipe_animation(frames)


# ---------------- Main Function ----------------
if __name__ == "__main__":
    # Example points and connections
    point_dict = {
        'A': [0, 0, 0],
        'B': [0, 1, 0],
        'C': [1, 1, 0],
        'D': [1, 0, 0],
        'E': [0.5, 1, 0.2],
        'G': [0.25, 0.75, 0.8]
    }

    group_connections = [['A', 'C', 'E','G'],['D','B']]
    frames, curves = generate_pipe_paths(point_dict, group_connections, pipe_radius=0.15)
    point_names = sum(group_connections, [])
    points = [point_dict[name] for name in point_names]  
    visualize_pipe_animation(frames, points=points)
