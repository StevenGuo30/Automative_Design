import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def visualize_pipe_animation(frames, points=None, obb_list=None, failed_segments=None):
    """
    Visualize pipe path animation with optional OBBs and failed segments.

    Parameters:
        frames: list of successful pipe curves (each a Nx3 array)
        points: list of input 3D points
        obb_list: list of linkage OBBs (each with .corners and .center)
        failed_segments: list of pipe curves that failed collision check
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def draw_obb(ax, obb):
        corners = obb["corners"]
        faces = [
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

    # Ensure at least 1 frame to trigger animation
    total_frames = max(len(frames), 1)

    def update(frame_idx):
        ax.clear()

        if len(frames) == 0 and frame_idx == 0:
            print("[Warning] No successful paths. Showing failed segments and OBBs only.")

        for i in range(min(frame_idx + 1, len(frames))):
            curve_pts = np.array(frames[i])
            ax.plot(curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2],
                    'b-', linewidth=2, label='Pipe' if i == 0 else None)

        if failed_segments:
            for i, seg in enumerate(failed_segments):
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                        'k--', linewidth=1, label='Failed Segment' if i == 0 else None)

        if points is not None:
            pts = np.array(points)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c='red', s=50, marker='o', label='Input Points')
            for i, (x, y, z) in enumerate(pts):
                ax.text(x, y, z, f'{i}', fontsize=10, color='black')

        if obb_list:
            for obb in obb_list:
                draw_obb(ax, obb)

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
