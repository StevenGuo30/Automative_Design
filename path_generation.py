from scipy.interpolate import splprep, splev
import numpy as np

def generate_arc_3d(p0, p1, arc_height_ratio, num_points):
    p0, p1 = np.array(p0), np.array(p1)
    midpoint = (p0 + p1) / 2
    direction = p1 - p0
    direction /= np.linalg.norm(direction)
    rand_vec = np.random.randn(3)
    rand_vec -= rand_vec.dot(direction) * direction
    rand_vec /= np.linalg.norm(rand_vec)
    offset = arc_height_ratio * np.linalg.norm(p1 - p0)
    p_m = midpoint + rand_vec * offset
    a, b = p0 - p_m, p1 - p_m
    cross_ab = np.cross(a, b)
    center = p_m + np.cross((np.linalg.norm(b)**2 * a - np.linalg.norm(a)**2 * b), cross_ab) / (2 * np.linalg.norm(cross_ab)**2)
    r = np.linalg.norm(p0 - center)
    u = (p0 - center) / r
    v = np.cross(cross_ab, u); v /= np.linalg.norm(v)
    theta1 = np.arctan2(np.dot((p1 - center) / r, v), np.dot((p1 - center) / r, u))
    if theta1 < 0: theta1 += 2 * np.pi
    ts = np.linspace(0, theta1, num_points)
    arc = np.array([center + r * (np.cos(t) * u + np.sin(t) * v) for t in ts])
    end_tangent = arc[-1] - arc[-2]; end_tangent /= np.linalg.norm(end_tangent)
    return arc, end_tangent

def generate_bezier(p0, p1, tangent, scale=0.3, num_points=50):
    p0, p1 = np.array(p0), np.array(p1)
    ctrl = p0 + scale * np.linalg.norm(p1 - p0) * tangent / np.linalg.norm(tangent)
    ts = np.linspace(0, 1, num_points)
    curve = np.array([(1 - t)**2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1 for t in ts])
    end_tangent = curve[-1] - curve[-2]; end_tangent /= np.linalg.norm(end_tangent)
    return curve, end_tangent

def interpolate_path(points, arc_height_ratio, num_points):
    """
    Generate smooth curve through given 3D points.
    Use B-spline if possible, otherwise fall back to arc + Bézier strategy.

    Parameters:
    - points: list of 3D points
    - arc_height_ratio: smoothing factor (for B-spline or arc height)
    - num_points: number of output sample points

    Returns:
    - np.ndarray of shape (num_points, 3)
    """
    points = [np.array(p) for p in points]

    if len(points) < 2:
        raise ValueError("At least two points are required.")

    # Try B-spline interpolation
    try:
        point_array = np.array(points)
        tck, u = splprep([point_array[:, 0], point_array[:, 1], point_array[:, 2]], s=arc_height_ratio)
        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine, z_fine = splev(u_fine, tck)
        return np.vstack([x_fine, y_fine, z_fine]).T
    except Exception as e:
        print(f"B-spline interpolation failed, falling back to arc+bezier: {e}")

    # Fallback: arc + Bézier
    from path_generation import generate_arc_3d, generate_bezier

    segments = []
    arc, tangent = generate_arc_3d(points[0], points[1], arc_height_ratio, num_points)
    segments.append(arc)

    for i in range(1, len(points) - 1):
        bezier, tangent = generate_bezier(points[i], points[i + 1], tangent, scale=0.33, num_points=num_points)
        segments.append(bezier[1:])  # Skip duplicated start point

    return np.vstack(segments)