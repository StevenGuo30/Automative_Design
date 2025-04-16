from typing import Any, TypeAlias, Protocol, TypedDict
from collections import defaultdict
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
from collision import (
    is_self_collision_by_sweep,
    is_curve_collision_by_sweep,
    generate_swept_pipe,
)
from rrt_planner import rrt_path, smooth_path_with_bspline, obb_list_to_pointcloud
from visualization import (
    visualize_pipe_animation,
    plot_problem,
    plot_curves,
    plot_lines,
)
from input_interface import input_interface, get_all_obbs
from scipy.spatial import cKDTree

from geometry.protocol import Point3D, Point3Ds, BOX, Simple1DGeometry, Node
from geometry import (
    Line,
    Curve,
    nearest_distance_between_geometry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_splines_to_json(
    spline_list: list[np.ndarray], output_path: str, num_sample_points: int = 300
) -> None:
    serialized = []
    for spline in spline_list:
        t = np.linspace(0, 1, spline.shape[0])
        t_sample = np.linspace(0, 1, num_sample_points)
        resampled = np.vstack(
            [
                np.interp(t_sample, t, spline[:, 0]),
                np.interp(t_sample, t, spline[:, 1]),
                np.interp(t_sample, t, spline[:, 2]),
            ]
        ).T
        serialized.append(resampled.tolist())
    with open(output_path, "w") as f:
        json.dump(serialized, f, indent=2)


def generate_pipe_paths(
    point_dict: dict[str, list[float]],
    group_names: list[list[str]],
    pipe_radius: float,
    obb_list: list[dict[str, Any]] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    base_points: dict[str, np.ndarray] = {
        name: np.array(coord) for name, coord in point_dict.items()
    }
    frames: list[np.ndarray] = []
    all_success_pipes: list[dict[str, np.ndarray]] = []
    failed_segments: list[np.ndarray] = []

    for group_idx, group in enumerate(group_names):
        print(f"\n[Generate_path]Processing group {group_idx+1}: {group}")
        group_points = [base_points[name] for name in group]
        success = False

        print(f"[Generate_path]Generating RRT path for group {group_idx+1}...")

        pipe_cloud = (
            np.vstack([p["curve"] for p in all_success_pipes])
            if all_success_pipes
            else np.empty((0, 3))
        )
        obb_cloud = obb_list_to_pointcloud(obb_list) if obb_list else np.empty((0, 3))
        obstacle_points = np.vstack([pipe_cloud, obb_cloud])

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
                max_iters=5000,
                step_size=0.7,
                goal_sample_rate=0.2,
                safe_radius=pipe_radius * 1.2,
                pipe_radius=pipe_radius,
                obb_list=obb_list,
            )

            if rrt_result is None:
                print(
                    f"[Generate_path]For Group {group_idx+1},RRT failed between point {i} and {i+1}, aborting group."
                )
                break

            rough_curve = smooth_path_with_bspline(
                rrt_result, num_points=1000
            )  # roughly smoothed
            smoothed_segment = smooth_path_with_bspline(rough_curve, num_points=3000)
            swept = generate_swept_pipe(smoothed_segment, pipe_radius)

            if is_self_collision_by_sweep(
                swept, obb_list=obb_list, entry_points=[start, goal]
            ):
                print(
                    f"[Generate_path]Smoothed segment between {i} and {i+1} collides, aborting group."
                )
                failed_segments.append(smoothed_segment)
                break

            if (
                i > 0
            ):  # Remove the first point of segment so it doesn't overlap with the last point of the previou segment
                smoothed_segment = smoothed_segment[1:]

            curve_segments.append(smoothed_segment)

        else:
            print(f"[Generate_path]Group {group_idx+1} RRT path generation succeeded.")
            full_curve = np.vstack(curve_segments)
            globally_smoothed_curve = smooth_path_with_bspline(
                full_curve, num_points=300
            )
            swept = generate_swept_pipe(globally_smoothed_curve, pipe_radius)

            if is_self_collision_by_sweep(
                swept,
                obb_list=obb_list,
                entry_points=[group_points[0], group_points[-1]],
            ):
                print(
                    f"[Generate_path]Group {group_idx+1} global smoothed curve has self-collision or OBB collision."
                )
                failed_segments.append(globally_smoothed_curve)
            elif any(
                is_curve_collision_by_sweep(swept, pipe["swept"], obb_list=obb_list)
                for pipe in all_success_pipes
            ):
                print(
                    f"[Generate_path]Group {group_idx+1} global smoothed curve collides with others or OBB."
                )
                failed_segments.append(globally_smoothed_curve)
            else:
                all_success_pipes.append(
                    {"curve": globally_smoothed_curve, "swept": swept}
                )
                frames.append(globally_smoothed_curve.copy())
                print(f"[Generate_path]Group {group_idx+1} succeeded with RRT.")
                success = True

        if not success:
            print(
                f"[Generate_path]Group {group_idx+1} failed after RRT path generation."
            )

    return frames, [pipe["curve"] for pipe in all_success_pipes], failed_segments


# ####### Changes: #######


def create_offset(node: Node, offset_mm: float = 5.0) -> Node:
    return {
        "coordinates": node["coordinates"] + offset_mm * node["direction"],
        "direction": node["direction"],
    }


def is_geometry_in_box(geometry: Simple1DGeometry, box: BOX) -> bool:
    points = geometry.sample()
    x_min, x_max, y_min, y_max, z_min, z_max = box
    return (
        (x_min > points[:, 0]).any()
        | (x_max < points[:, 0]).any()
        | (y_min > points[:, 1]).any()
        | (y_max < points[:, 1]).any()
        | (z_min > points[:, 2]).any()
        | (z_max < points[:, 2]).any()
    )


from scipy.optimize import minimize, NonlinearConstraint


def get_optimal_curve_avoiding_collision(
    input_node: Node,
    target_node: Node,
    collision_lines: list[Line],
    collision_curves: dict[int, list[Curve]],
    bbox: BOX,  # TODO
) -> tuple[Curve, float]:
    input_point = input_node["coordinates"]
    input_direction = input_node["direction"]
    target_point = target_node["coordinates"]
    target_direction = target_node["direction"]
    curves_flat = list(itertools.chain(*collision_curves.values()))

    # extent: input_point + extent * input_direction
    def objective(
        params: tuple[float, float, float],
    ):
        direction_scale_input, direction_scale_output, input_extent = params
        curve = Curve(
            input_point,
            target_point,
            input_direction,
            target_direction,
            direction_scale_input,
            direction_scale_output,
            input_extent,
        )
        return (
            curve.energy()
            + 1e-2 * input_extent
            + 5e-2 * direction_scale_input
            + 5e-2 * direction_scale_output
        )

    def constraint_penetration_count(params: tuple[float, float, float]) -> float:
        direction_scale_input, direction_scale_output, input_extent = params
        curve = Curve(
            input_point,
            target_point,
            input_direction,
            target_direction,
            direction_scale_input,
            direction_scale_output,
            input_extent,
        )
        penetration_count = 0
        # Penetration against other lines and curves
        for geom in collision_lines + curves_flat:
            if nearest_distance_between_geometry(curve, geom) < pipe_radius:
                penetration_count += 1
        # Penetration against bounding box
        if is_geometry_in_box(curve, bbox):
            penetration_count += 1
        return penetration_count

    res = minimize(
        objective,
        x0=[0.0, 0.0, 0.0],
        bounds=[(10.0, 1e2), (10.0, 1e2), (0.0, 1e1)],
        constraints=[
            NonlinearConstraint(constraint_penetration_count, -np.inf, 1e-2),  # zero
        ],
        options={"disp": True},  # "maxiter": 100,
    )
    direction_scale_input, direction_scale_output, input_extent = res.x

    new_curve = Curve(
        input_point,
        target_point,
        input_direction,
        target_direction,
        direction_scale_input,
        direction_scale_output,
        input_extent,
    )

    # Log curve creation details and energy
    curve_energy = new_curve.energy()
    curve_length = new_curve.length()
    logger.info(f"From {input_point} to {target_point}")
    logger.info(f"Direction start: {input_direction}, end: {target_direction}")
    logger.info(f"{input_extent=}, {direction_scale_input=}, {direction_scale_output=}")
    logger.info(f"Energy = {curve_energy:.4f}, Length = {curve_length:.4f}")

    return new_curve, input_extent


def find_bounding_box(points) -> BOX:
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    z_min, z_max = min(z), max(z)
    return x_min, x_max, y_min, y_max, z_min, z_max


def ax_zoom_fit(ax, bbox: BOX):
    """Optimize the zoom"""
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_zlim(bbox[4], bbox[5])


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "test_points.json")
    assert os.path.exists(json_path), f"{json_path} not found"

    # Problem definition
    with open(json_path, "r") as f:
        data = json.load(f)
    pipe_radius: float = data["pipe_radius"] * data["pipe_radius_safety_factor"]
    nodes = data["nodes"]
    for node in nodes.values():
        node["coordinates"] = np.array(node["coordinates"])
        node["direction"] = np.array(node["direction"])
    inputs = data["inputs"]
    groups = data["groups"]
    assert len(inputs) == len(groups), "inputs and groups must have the same length"
    # obb_list = get_all_obbs(list(point_dict.values()))

    bbox = find_bounding_box([node["coordinates"] for node in nodes.values()])

    # Problem visualization
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection="3d")
    # for gidx, group in enumerate(groups):
    #     points_array = np.array([nodes[name]["coordinates"] for name in group])
    #     directions_array = np.array([nodes[name]["direction"] for name in group])
    #     plot_problem(
    #         ax,
    #         points_array,
    #         directions_array,
    #         group,
    #         f"group {gidx}",
    #     )
    # # set equal aspect ratio
    # ax.set_box_aspect([1.0, 1.0, 1.0])
    # plt.tight_layout()
    # plt.show()

    lines = []
    lines_extended = []
    curves: dict[int, list[Curve]] = defaultdict(list)

    # Create offset lines to avoid collision with initial block
    offset_points = {}
    for node_name, node in nodes.items():
        offset_point = create_offset(node)
        lines.append(Line(node["coordinates"], offset_point["coordinates"]))
        offset_points[node_name] = offset_point

    # Create connections from inputs to nodes
    for gidx, group in enumerate(groups):
        input_node = offset_points[inputs[gidx]]
        target_nodes = [offset_points[name] for name in group if name not in inputs]
        max_input_extent = 0.0
        for target_node in target_nodes:
            curve, input_extent = get_optimal_curve_avoiding_collision(
                input_node, target_node, lines, curves, bbox
            )
            max_input_extent = max(max_input_extent, input_extent)
            curves[gidx].append(curve)
        line_extension = (
            input_node["coordinates"] + max_input_extent * input_node["direction"]
        )
        lines_extended.append(Line(input_node["coordinates"], line_extension))

    # Draw solution
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    [print(l) for l in lines]
    [print(l) for l in lines_extended]
    plot_lines(ax, lines, pipe_radius, color="black")
    plot_lines(ax, lines_extended, pipe_radius, color="red")
    for gidx, group in enumerate(groups):
        points_array = np.array([nodes[name]["coordinates"] for name in group])
        offset_points_array = np.array(
            [offset_points[name]["coordinates"] for name in group]
        )
        directions_array = np.array([nodes[name]["direction"] for name in group])
        curve_group = curves[gidx]
        color = plot_problem(
            ax,
            points_array,
            directions_array,
            group,
            f"group {gidx}",
        )
        ax.scatter(
            offset_points_array[:, 0],
            offset_points_array[:, 1],
            offset_points_array[:, 2],
            color=color,
            alpha=0.5,
        )
        plot_curves(ax, curve_group, f"group {gidx}", pipe_radius, color=color)
    ax_zoom_fit(ax, bbox)
    plt.tight_layout()
    plt.show()
    breakpoint()

    sys.exit()
    frames, curves, failed_segments = generate_pipe_paths(
        point_dict, group_names, pipe_radius, obb_list=obb_list
    )

    save_path = os.path.join(script_dir, "exported_splines.json")
    save_splines_to_json(curves, save_path, num_sample_points=300)
    print(f"Saved {len(curves)} curves to {save_path}")

    # points = [point_dict[name] for group in group_names for name in group]
    visualize_pipe_animation(
        frames,
        point_dict=point_dict,
        group_names=group_names,
        obb_list=obb_list,
        failed_segments=failed_segments,
    )
