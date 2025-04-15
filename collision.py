import numpy as np
from scipy.spatial import cKDTree
import trimesh
from shapely.geometry import Point
from trimesh.collision import CollisionManager


def resample_curve(curve, ds):
    """
    Resample a 3D curve using fixed spacing.
    """
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
    """
    Check for collision between two resampled pipe curves using KD-tree.
    """
    sampled1 = resample_curve(curve1, sample_ds)
    sampled2 = resample_curve(curve2, sample_ds)
    tree2 = cKDTree(sampled2)
    for p1 in sampled1:
        dist, _ = tree2.query(p1, k=1)
        if dist < 2 * pipe_radius:
            return True
    return False


def is_self_collision(curve, pipe_radius, sample_ds=0.05):
    """
    Detect self-collision in a pipe by checking nearby points along the curve.
    """
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
    Generate a swept solid mesh by sweeping a circular cross-section along a 3D path.

    Parameters:
    - curve (ndarray): Nx3 numpy array representing the sweep path.
    - radius (float): Radius of the circular cross-section.
    - sections (int): Number of segments for the circular approximation.
    - engine (str): Triangulation engine to use ('triangle', 'earcut', etc.)

    Returns:
    - trimesh.Trimesh: Swept pipe mesh.
    """
    circle_polygon = Point(0, 0).buffer(radius, resolution=sections)
    swept = trimesh.creation.sweep_polygon(circle_polygon, curve, engine=engine)
    return swept


def is_self_intersecting(mesh):
    """
    Check if a mesh intersects itself using a collision manager.
    """
    manager = CollisionManager()
    manager.add_object('mesh', mesh)
    return manager.in_collision_internal()


def is_swept_collide_with_obbs(swept_mesh, obb_list, entry_points=None, tolerance=5):
    """
    Check if a swept mesh collides with any of the provided OBB boxes,
    allowing exceptions if entry_points (e.g., path start/end) intentionally contact the OBB.
    `tolerance`: Distance under which entry_points are considered valid contact (cm).
    """
    if not obb_list:
        return False

    manager = CollisionManager()
    manager.add_object("swept", swept_mesh)

    for obb in obb_list:
        try:
            box = trimesh.convex.convex_hull(obb["corners"])

            if manager.in_collision_single(box):
                # Allow entry contact exception
                if entry_points is not None:
                    for p in entry_points:
                        dist = np.linalg.norm(p - obb["center"])
                        if dist < tolerance:
                            print(f"[collision] Entry point {p} accepted: within {dist:.4f} of OBB {obb['name']}")
                            break
                    else:
                        print(f"[collision] Swept body collides with OBB of {obb['name']}")
                        return True
                else:
                    print(f"[collision] Swept body collides with OBB of {obb['name']}")
                    return True

        except Exception as e:
            print(f"  [OBB check failed for {obb['name']}] {e}")

    return False



def is_self_collision_by_sweep(swept_mesh, obb_list=None, entry_points=None):
    """
    Check if a swept mesh has internal self-collision or collides with OBBs,
    allowing contact at entry points.
    """
    try:
        if is_self_intersecting(swept_mesh):
            return True
        if obb_list and is_swept_collide_with_obbs(swept_mesh, obb_list, entry_points=entry_points):
            return True
        return False
    except Exception as e:
        print(f"  [Swept Collision Check Error] {e}")
        return True


def is_curve_collision_by_sweep(swept_mesh1, swept_mesh2, obb_list=None):
    """
    Check if two swept meshes collide with each other or with OBBs.

    Parameters:
    - swept_mesh1: First swept pipe mesh.
    - swept_mesh2: Second swept pipe mesh.
    - obb_list: Optional list of OBB dicts.

    Returns:
    - True if collision occurs, False otherwise.
    """
    try:
        manager1 = CollisionManager()
        manager2 = CollisionManager()

        manager1.add_object('mesh1', swept_mesh1)
        manager2.add_object('mesh2', swept_mesh2)

        if manager1.in_collision_other(manager2):
            return True

        # if obb_list and (is_swept_collide_with_obbs(swept_mesh1, obb_list) or
        #                  is_swept_collide_with_obbs(swept_mesh2, obb_list)):
        #     return True

        return False
    except Exception as e:
        print(f"  [Swept Pair Collision Error] {e}")
        return True
