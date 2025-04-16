import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .protocol import Point3D, Point3Ds


class Curve:
    def __init__(
        self,
        point_a: Point3D,
        point_b: Point3D,
        direction_a: Point3D,
        direction_b: Point3D,
        direction_scale_input: float = 1.0,
        direction_scale_output: float = 1.0,
        input_extent: float = 0.0,
    ) -> None:
        self.point_a = np.asarray(point_a, dtype=float)
        self.point_b = np.asarray(point_b, dtype=float)
        self.direction_a = np.asarray(direction_a, dtype=float) * direction_scale_input
        self.direction_b = (
            np.asarray(direction_b, dtype=float) * direction_scale_output
        )

        # Extension
        self.point_a = self.point_a + input_extent * self.direction_a
        self._cached_samples: dict[int, Point3Ds] = {}

    def sample(self, discretization: int = 100) -> Point3Ds:
        if discretization in self._cached_samples:
            return self._cached_samples[discretization]
        t = np.linspace(0, 1, discretization)
        self._cached_samples[discretization] = self.evaluate(t)
        return self.evaluate(t)

    def evaluate(self, t: float | np.ndarray) -> Point3D:
        t = np.asarray(t)
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return (
            h00[:, None] * self.point_a[None, :]
            + h10[:, None] * self.direction_a[None, :]
            + h01[:, None] * self.point_b[None, :]
            + h11[:, None] * self.direction_b[None, :]
        )

    def length(self) -> float:
        points = self.evaluate(np.linspace(0, 1, 100))
        return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=-1))

    def plot(self, ax: Axes3D, color: str, pipe_radius: float) -> None:
        t = np.linspace(0, 1, 100)
        ax.plot(
            self.evaluate(t)[:, 0],
            self.evaluate(t)[:, 1],
            self.evaluate(t)[:, 2],
            color=color,
            linewidth=2,
        )

    def _bending_energy(self, num_points: int = 100) -> float:
        """
        Compute an approximation of the bending energy,
        E_bend = ∫ (curvature)^2 ds.
        """
        t_vals = np.linspace(0, 1, num_points)
        pts = self.evaluate(t_vals)
        # Finite differences for first and second derivatives.
        dt = t_vals[1] - t_vals[0]
        d_pts = np.gradient(pts, dt, axis=0)
        dd_pts = np.gradient(d_pts, dt, axis=0)

        # Compute speed, curvature.
        speed = np.linalg.norm(d_pts, axis=1)  # ||gamma'(t)||
        cross_prod = np.cross(d_pts, dd_pts)
        # To avoid division by zero, add a small epsilon.
        epsilon = 1e-8
        curvature = np.linalg.norm(cross_prod, axis=1) / (speed**3 + epsilon)

        # ds = speed * dt, so energy approximated by sum( curvature^2 * ds ).
        bending_energy = np.sum((curvature**2) * speed * dt)
        return bending_energy

    def _twisting_energy(self, num_points: int = 100) -> float:
        """
        Approximate the twisting energy.

        The idea is:
        1. Build a material director along the curve by parallel transporting
           the initial director (projected onto the tangent plane).
        2. Compare the final transported director with the projection of the
           desired director d1 in the tangent plane at the end.
        3. If the net angle difference is Δθ and the total arc length is L, then
           for a uniform twist rate ω = Δθ / L, the twist energy is approximated by
                 E_twist = L * ω^2 = (Δθ)^2 / L.
        """
        # Sample points along curve.
        t_vals = np.linspace(0, 1, num_points)
        pts = self.evaluate(t_vals)
        dt = t_vals[1] - t_vals[0]
        d_pts = np.gradient(pts, dt, axis=0)
        speed = np.linalg.norm(d_pts, axis=1)
        # Total arc length.
        L = np.sum(speed * dt)

        # Compute unit tangents along the curve.
        T = d_pts / (speed[:, None] + 1e-8)

        # Build the transported director field.
        # At t=0, project d0 onto the normal plane of T[0]:
        D = np.zeros_like(pts)
        d0_proj = self.direction_a - np.dot(self.direction_a, T[0]) * T[0]
        if np.linalg.norm(d0_proj) < 1e-8:
            # In case the provided director is aligned with the tangent.
            # Choose an arbitrary perpendicular direction.
            d0_proj = np.cross(T[0], np.array([1, 0, 0]))
            if np.linalg.norm(d0_proj) < 1e-8:
                d0_proj = np.cross(T[0], np.array([0, 1, 0]))
        D[0] = d0_proj / np.linalg.norm(d0_proj)

        # Parallel transport: for i=1 to N, update D[i] from D[i-1].
        for i in range(1, num_points):
            # Compute rotation needed from T[i-1] to T[i].
            # The axis is n = T[i-1] x T[i]
            n = np.cross(T[i - 1], T[i])
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-8:
                # If tangents are nearly aligned, no rotation.
                D[i] = D[i - 1]
            else:
                n = n / norm_n
                # Angle between the two tangents.
                dot_val = np.clip(np.dot(T[i - 1], T[i]), -1.0, 1.0)
                theta = np.arccos(dot_val)
                # Rodriguez rotation formula.
                D[i] = (
                    D[i - 1] * np.cos(theta)
                    + np.cross(n, D[i - 1]) * np.sin(theta)
                    + n * np.dot(n, D[i - 1]) * (1 - np.cos(theta))
                )
                # After rotation, project D[i] to the normal plane of T[i].
                D[i] = D[i] - np.dot(D[i], T[i]) * T[i]
                D[i] = D[i] / (np.linalg.norm(D[i]) + 1e-8)

        # At t = 1, the desired (target) director is d1 projected onto the tangent plane.
        d1_proj = self.direction_b - np.dot(self.direction_b, T[-1]) * T[-1]
        if np.linalg.norm(d1_proj) < 1e-8:
            # In case the provided director is aligned with the tangent.
            d1_proj = np.cross(T[-1], np.array([1, 0, 0]))
            if np.linalg.norm(d1_proj) < 1e-8:
                d1_proj = np.cross(T[-1], np.array([0, 1, 0]))
        d1_proj = d1_proj / np.linalg.norm(d1_proj)

        # Compute the net angle difference between the transported director D[-1] and d1_proj.
        # The angle between two unit vectors is given by arccos of their dot product.
        dotD = np.clip(np.dot(D[-1], d1_proj), -1.0, 1.0)
        delta_theta = np.arccos(dotD)

        # Assuming uniform twist, twist rate = delta_theta / L, so
        # twisting energy approximated by (delta_theta)^2 / L.
        twisting_energy = (delta_theta**2) / (L + 1e-8)
        return twisting_energy

    def energy(self, num_points=100, alpha=0.8) -> float:
        """
        Compute the total energy, a sum of bending and twisting energies:
           E_total = 0.5*E_bend + 0.5*alpha*E_twist.
        You may adjust the weights alpha.
        """
        E_bend = self._bending_energy(num_points)
        E_twist = self._twisting_energy(num_points)
        return 0.5 * E_bend + 0.5 * alpha * E_twist
