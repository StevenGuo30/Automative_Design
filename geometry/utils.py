import numpy as np

from .protocol import Point3D, BOX, Simple1DGeometry


def nearest_distance_between_geometry(
    geometry_a: Simple1DGeometry,
    geometry_b: Simple1DGeometry,
    discretization: int = 100,
) -> float:
    """Calculate the nearest distance between two geometries."""
    points_a = geometry_a.sample(discretization)
    points_b = geometry_b.sample(discretization)
    distances = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=-1)
    return float(np.min(distances))
