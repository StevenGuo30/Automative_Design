import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .protocol import Point3D, Point3Ds


class Line:
    def __init__(self, start: Point3D, end: Point3D):
        self.start = start
        self.end = end
        self._cached_samples: dict[int, Point3Ds] = {}

    def __repr__(self):
        return f"Line(start={self.start}, end={self.end})"

    def sample(self, discretization: int = 100) -> Point3Ds:
        if discretization in self._cached_samples:
            return self._cached_samples[discretization]
        t = np.linspace(0, 1, discretization)
        self._cached_samples[discretization] = self.evaluate(t)
        return self._cached_samples[discretization]

    def evaluate(self, t):
        t = np.asarray(t)
        return self.start[None, :] + t[:, None] * (self.end - self.start)[None, :]

    def energy(self) -> float:
        return 0.0

    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    def plot(self, ax: Axes3D, color: str, pipe_radius: float) -> None:
        ax.plot(
            [self.start[0], self.end[0]],
            [self.start[1], self.end[1]],
            [self.start[2], self.end[2]],
            color=color,
            linestyle="--",
        )
