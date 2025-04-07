
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def visualize_pipe_animation(frames, points=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        for i in range(frame_idx + 1):
            curve_pts = np.array(frames[i])
            ax.plot(curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2], 'b-', linewidth=2)

        if points is not None:
            pts = np.array(points)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=50, marker='o', label='Input Points')
            for i, (x, y, z) in enumerate(pts):
                ax.text(x, y, z, f'{i}', fontsize=10, color='black')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Step {frame_idx+1}/{len(frames)}")
        ax.grid(True)
        ax.legend()

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1500, repeat=False)
    plt.tight_layout()
    plt.show()
