from typing import Any

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


# Draw OBBs
def draw_obb(ax: Axes3D, corners: np.ndarray) -> None:
    faces: list[list[np.ndarray]] = [
        [corners[i] for i in [0, 1, 3, 2]],
        [corners[i] for i in [4, 5, 7, 6]],
        [corners[i] for i in [0, 1, 5, 4]],
        [corners[i] for i in [2, 3, 7, 6]],
        [corners[i] for i in [1, 3, 7, 5]],
        [corners[i] for i in [0, 2, 6, 4]],
    ]
    face = Poly3DCollection(faces, alpha=0.25, linewidths=0.5)
    face.set_facecolor("gray")
    ax.add_collection3d(face)


def visualize_pipe_animation(
    frames, point_dict=None, group_connections=None, obb_list=None, failed_segments=None
):
    """
    Visualize pipe path animation with optional OBBs and failed segments.

    Parameters:
        frames: list of successful pipe curves (each a Nx3 array)
        points: list of input 3D points
        obb_list: list of linkage OBBs (each with .corners and .center)
        failed_segments: list of pipe curves that failed collision check
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Ensure at least 1 frame to trigger animation
    total_frames = max(len(frames), 1)

    def update(frame_idx):
        ax.clear()

        if len(frames) == 0 and frame_idx == 0:
            print(
                "[Warning] No successful paths. Showing failed segments and OBBs only."
            )

        for i in range(min(frame_idx + 1, len(frames))):
            curve_pts = np.array(frames[i])
            ax.plot(
                curve_pts[:, 0],
                curve_pts[:, 1],
                curve_pts[:, 2],
                "b-",
                linewidth=2,
                label="Pipe" if i == 0 else None,
            )

        if failed_segments:
            for i, seg in enumerate(failed_segments):
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    seg[:, 2],
                    "k--",
                    linewidth=1,
                    label="Failed Segment" if i == 0 else None,
                )
        cnt = 0
        # group_points = [point_dict[name] for group in group_connections for name in group]

        for group in group_connections:
            pts = np.array([point_dict[name] for name in group])
            if cnt == 0:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    c="red",
                    s=50,
                    marker="o",
                    label="Input Points",
                )
                for i, (x, y, z) in enumerate(pts):
                    ax.text(x, y, z, f"{i}", fontsize=10, color="black")
            elif cnt == 1:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    c="orange",
                    s=50,
                    marker="o",
                    label="Input Points",
                )
                for i, (x, y, z) in enumerate(pts):
                    ax.text(x, y, z, f"{i}", fontsize=10, color="black")
            cnt += 1

        if obb_list:
            for obb in obb_list:
                draw_obb(ax, obb["corners"])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        title = f"Step {frame_idx+1}/{len(frames)}"
        if len(frames) == 0:
            title += " [No successful path]"
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    ani = FuncAnimation(fig, update, frames=total_frames, interval=1500, repeat=False)
    plt.tight_layout()
    plt.show()


def plot_problem(
    ax: Axes3D,
    points: np.ndarray,
    directions: np.ndarray,
    group_names: list[str],
    label: str,
) -> list[float]:
    """
    Visualize the problem statement with matplotlib before pipe generation.

    Parameters:
        points: 3D coordinates [N, 3]
        directions: directions [N, 3]
        group_names: point names
    """

    s = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=70,
        marker="o",
        label=label,
    )

    # Add director as arrow
    quiver_scale = 0.5
    ax.quiver(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        directions[:, 0] * quiver_scale,
        directions[:, 1] * quiver_scale,
        directions[:, 2] * quiver_scale,
        color=s.get_facecolor(),
    )

    # Add labels
    for i, point_name in enumerate(group_names):
        ax.text(
            *points[i],
            point_name,
            fontsize=10,
            color="black",
        )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Problem Definition: Connection Points and Obstacles")
    ax.legend()

    return s.get_facecolor()


def plot_lines(
    ax: Axes3D, lines: list, pipe_radius: float, color: str = "gray"
) -> None:
    """
    Plot a list of line objects on the given 3D axes.

    Parameters:
        ax: The matplotlib 3D axes
        lines: List of Line objects
        pipe_radius: Radius of the pipe for visualization purposes
    """
    for i, line in enumerate(lines):
        line.plot(ax, color, pipe_radius)


def plot_curves(
    ax: Axes3D, curves: list, label: str, pipe_radius: float, color=None
) -> None:
    """
    Plot a list of curve objects on the given 3D axes.

    Parameters:
        ax: The matplotlib 3D axes
        curves: List of Curve objects
        label: Label for the legend
        pipe_radius: Radius of the pipe for visualization purposes
        color: Color to use for all curves (optional)
    """
    for i, curve in enumerate(curves):
        curve.plot(ax, color if color is not None else "blue", pipe_radius)

    # If we have curves, add a legend entry for the first one
    if curves and label:
        # Add a dummy plot for legend purposes
        ax.plot(
            [],
            [],
            color=color if color is not None else "blue",
            linewidth=2,
            label=label,
        )
