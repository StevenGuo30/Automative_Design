import numpy as np
from scipy.spatial import cKDTree
import trimesh
from shapely.geometry import Point
from trimesh.collision import CollisionManager


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

def is_pipe_collision(curve1, curve2, pipe_radius, sample_ds=0.05):
    sampled1 = resample_curve(curve1, sample_ds)
    sampled2 = resample_curve(curve2, sample_ds)
    tree2 = cKDTree(sampled2)
    for p1 in sampled1:
        dist, _ = tree2.query(p1, k=1)
        if dist < 2 * pipe_radius:
            return True
    return False

def is_self_collision(curve, pipe_radius, sample_ds=0.05):
    sampled = resample_curve(curve, sample_ds)
    tree = cKDTree(sampled)
    for i, pt in enumerate(sampled):
        idxs = tree.query_ball_point(pt, r=2 * pipe_radius)
        for j in idxs:
            if abs(i - j) > 2:
                return True
    return False


def generate_swept_pipe(curve, radius=1.0, sections=32, engine='triangle'):
    """
    Generate a swept solid mesh by sweeping a circular polygon along a 3D path.

    Parameters:
    - curve (ndarray): Nx3 numpy array representing the sweep path.
    - radius (float): Radius of the circular cross-section.
    - sections (int): Number of segments for the circular approximation.
    - engine (str): Triangulation engine to use ('triangle', 'earcut', etc.)

    Returns:
    - trimesh.Trimesh: Swept pipe mesh.
    """
    # Use shapely to create a 2D circular cross-section
    circle_polygon = Point(0, 0).buffer(radius, resolution=sections)  # shapely.geometry.Polygon

    # Use trimesh to sweep along the curve
    swept = trimesh.creation.sweep_polygon(circle_polygon, curve, engine=engine)

    return swept

def is_self_intersecting(mesh):
    # 创建碰撞管理器并添加网格
    manager = CollisionManager()
    manager.add_object('mesh', mesh)

    # 检查网格与自身的碰撞
    return manager.in_collision_internal()

def is_self_collision_by_sweep(swept_mesh):
    try:
        return is_self_intersecting(swept_mesh)
    except Exception as e:
        print(f"  [Swept Collision Check Error] {e}")
        return True
    

def is_curve_collision_by_sweep(swept_mesh1, swept_mesh2):
    try:
        # 创建碰撞管理器
        manager1 = CollisionManager()
        manager2 = CollisionManager()
        
        # 将第一个网格添加到第一个管理器
        manager1.add_object('mesh1', swept_mesh1)
        # 将第二个网格添加到第二个管理器
        manager2.add_object('mesh2', swept_mesh2)
        
        # 检查两个管理器中的对象是否发生碰撞
        collision = manager1.in_collision_other(manager2)
        return collision
    except Exception as e:
        print(f"  [Swept Pair Collision Error] {e}")
        return True
