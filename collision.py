
import numpy as np
from scipy.spatial import cKDTree

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
