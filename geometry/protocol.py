from typing import Any, Protocol, TypeAlias, TypedDict
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

Point3D: TypeAlias = np.ndarray
Point3Ds: TypeAlias = np.ndarray
BOX: TypeAlias = tuple[float, float, float, float, float, float]


class Node(TypedDict):
    coordinates: Point3D
    direction: Point3D


class Simple1DGeometry(Protocol):
    def evaluate(self, t: float | np.ndarray) -> Point3D: ...
    def sample(self, discretization: int = 100) -> Point3Ds: ...
    def energy(self, *args: Any, **kwargs: Any) -> float: ...
    def length(self) -> float: ...
    def plot(self, ax: Axes3D, color: str, pipe_radius: float) -> None: ...
