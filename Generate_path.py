import os
import json
import numpy as np

from path_generation import interpolate_path
from collision import is_pipe_collision, is_self_collision, is_self_collision_by_sweep, is_curve_collision_by_sweep, generate_swept_pipe
from rrt_planner import rrt_path, smooth_path_with_bspline, is_path_collision
from visualization import visualize_pipe_animation
from input_interface import input_interface
from scipy.spatial import cKDTree

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
    frames = []
    all_success_pipes = []

    for group_idx, group in enumerate(group_connections):
        print(f"\nProcessing group {group_idx+1}: {group}")
        group_points = [base_points[name] for name in group]
        success = False

        print(f"Generating RRT path for group {group_idx+1}...")

        obstacle_points = np.vstack([p['curve'] for p in all_success_pipes]) if all_success_pipes else np.empty((0, 3))
        all_points = np.vstack(list(base_points.values()))
        margin = 0.5
        x_limits = (all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        y_limits = (all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        z_limits = (all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

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
                print(f"  RRT failed between point {i} and {i+1}, aborting group.")
                break

            smoothed_segment = smooth_path_with_bspline(rrt_result, num_points=100)

            swept = generate_swept_pipe(smoothed_segment, pipe_radius)
            if is_self_collision_by_sweep(swept):
                print(f"  Smoothed segment between {i} and {i+1} collides, aborting group.")
                break

            if i > 0:
                smoothed_segment = smoothed_segment[1:]

            curve_segments.append(smoothed_segment)

        else:
            full_curve = np.vstack(curve_segments)
            globally_smoothed_curve = smooth_path_with_bspline(full_curve, num_points=300)
            swept = generate_swept_pipe(globally_smoothed_curve, pipe_radius)

            if is_self_collision_by_sweep(swept):
                print(f"  Global smoothed curve has self-collision.")
            elif any(is_curve_collision_by_sweep(swept, pipe['swept']) for pipe in all_success_pipes):
                print(f"  Global smoothed curve collides with others.")
            else:
                all_success_pipes.append({'curve': globally_smoothed_curve, 'swept': swept})
                frames.append(globally_smoothed_curve.copy())
                print(f"Group {group_idx+1} succeeded with RRT.")
                success = True

        if not success:
            print(f"Group {group_idx+1} failed after RRT path generation.")

    return frames, [pipe['curve'] for pipe in all_success_pipes]



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "paired_points.json")

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
    sample_ds = 0.005

    frames, curves = generate_pipe_paths(point_dict, group_connections,
                                         arc_height_ratio, num_points,
                                         pipe_radius, sample_ds)

    save_path = os.path.join(script_dir, "exported_splines.json")
    save_splines_to_json(curves, save_path, num_sample_points=300)
    print(f"Saved {len(curves)} curves to {save_path}")

    points = [point_dict[name] for group in group_connections for name in group]
    visualize_pipe_animation(frames, points=points)
