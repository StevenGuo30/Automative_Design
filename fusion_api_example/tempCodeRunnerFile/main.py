
import os
import json
import numpy as np
import sys

# load upper of the upper directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
upper_upper_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(upper_upper_dir)

from path_generation import interpolate_path
from collision import is_pipe_collision, is_self_collision
from rrt_planner import rrt_path, smooth_path_with_bspline, is_path_collision
from visualization import visualize_pipe_animation
from input_interface import input_interface

def save_splines_to_json(spline_list, output_path, num_sample_points=300):
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

def generate_pipe_paths(point_dict, group_connections, arc_height_ratio, num_points,
                        pipe_radius, sample_ds, max_retries=10):
    base_points = {name: np.array(coord) for name, coord in point_dict.items()}
    curves = []
    frames = []
    all_success_curves = []

    for group_idx, group in enumerate(group_connections):
        print(f"Processing group {group_idx+1}: {group}")
        group_points = [base_points[name] for name in group]
        success = False

        for attempt in range(max_retries):
            try:
                curve = interpolate_path(group_points, arc_height_ratio, num_points)
                if any(is_pipe_collision(curve, c, pipe_radius, sample_ds) for c in all_success_curves):
                    continue
                if is_self_collision(curve, pipe_radius, sample_ds):
                    continue
                all_success_curves.append(curve)
                frames.append(curve.copy())
                success = True
                print(f"Group {group_idx+1} success at attempt {attempt+1}")
                break
            except Exception as e:
                print(f"Error in group {group_idx+1} attempt {attempt+1}: {e}")

        if not success:
            print(f"Group {group_idx+1} failed after all retries.")

    return frames, all_success_curves

if __name__ == "__main__":
    json_path = os.path.join(upper_upper_dir, "paired_points.json")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        pipe_radius = data["pipe_radius"]
        data = data["connections"]
        point_dict = {}
        group_connections = []
        for group in data:
            names = []
            for pt in group:
                name = pt["name"]
                coords = pt["coordinates"]
                point_dict[name] = [coords["x"], coords["y"], coords["z"]]
                names.append(name)
            group_connections.append(names)
        print(f"Loaded {len(data)} groups of points from {json_path}")
    else:
        print(f"{json_path} not found, generating points with input interface...")
        point_dict, group_connections = input_interface() #in input_interface function json file will be generated

    arc_height_ratio = 0.22
    num_points = 50
    sample_ds = 0.05

    frames, curves = generate_pipe_paths(point_dict, group_connections,
                                         arc_height_ratio, num_points,
                                         pipe_radius, sample_ds)

    save_path = os.path.join(script_dir, "exported_splines.json")
    save_splines_to_json(curves, save_path, num_sample_points=300)
    print(f"Saved {len(curves)} curves to {save_path}")

    points = [point_dict[name] for group in group_connections for name in group]
    visualize_pipe_animation(frames, points=points)
