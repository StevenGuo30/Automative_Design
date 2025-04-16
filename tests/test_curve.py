import pytest
import numpy as np
import matplotlib.pyplot as plt
from geometry.curve import Curve


class TestCurve:
    @pytest.fixture
    def simple_curve(self):
        # Create a simple curve from (0,0,0) to (1,1,1) with specific directions
        return Curve(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([1.0, 1.0, 1.0]),
            direction_a=np.array([1.0, 0.0, 0.0]),
            direction_b=np.array([0.0, 1.0, 0.0]),
        )

    def test_initialization(self, simple_curve):
        # Test basic initialization
        assert np.array_equal(simple_curve.point_a, np.array([0.0, 0.0, 0.0]))
        assert np.array_equal(simple_curve.point_b, np.array([1.0, 1.0, 1.0]))
        assert np.array_equal(simple_curve.direction_a, np.array([1.0, 0.0, 0.0]))
        assert np.array_equal(
            simple_curve.direction_b, np.array([0.0, -1.0, 0.0])
        )  # Note the sign flip in the constructor

    def test_initialization_with_input_extent(self):
        # Test initialization with input extent
        curve = Curve(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([1.0, 1.0, 1.0]),
            direction_a=np.array([1.0, 0.0, 0.0]),
            direction_b=np.array([0.0, 1.0, 0.0]),
            input_extent=0.5,
        )
        # The point_a should be shifted by input_extent * direction_a
        assert np.array_equal(curve.point_a, np.array([0.5, 0.0, 0.0]))

    def test_evaluate_single_point(self, simple_curve):
        # Evaluate at t=0 should equal point_a
        result = simple_curve.evaluate(0.0)
        assert np.allclose(result, np.array([0.0, 0.0, 0.0]))

        # Evaluate at t=1 should equal point_b
        result = simple_curve.evaluate(1.0)
        assert np.allclose(result, np.array([1.0, 1.0, 1.0]))

        # Test a point in the middle
        result = simple_curve.evaluate(0.5)
        # Not checking exact values as the Hermite interpolation is complex
        assert result.shape == (3,)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_evaluate_multiple_points(self, simple_curve):
        t_values = np.array([0.0, 0.5, 1.0])
        result = simple_curve.evaluate(t_values)

        # Shape should be (3, 3) - 3 points, each with 3 coordinates
        assert result.shape == (3, 3)

        # First point should be point_a
        assert np.allclose(result[0], np.array([0.0, 0.0, 0.0]))

        # Last point should be point_b
        assert np.allclose(result[2], np.array([1.0, 1.0, 1.0]))

    def test_sample(self, simple_curve):
        # Test with discretization of 5 points
        samples = simple_curve.sample(5)
        assert samples.shape == (5, 3)  # 5 points, each with 3 coordinates

        # First point should be point_a
        assert np.allclose(samples[0], simple_curve.point_a)

        # Last point should be point_b
        assert np.allclose(samples[-1], simple_curve.point_b)

        # Test caching behavior
        # Second call should return the cached result
        cached_samples = simple_curve.sample(5)
        assert id(cached_samples) == id(samples)  # Should return the exact same object

        # Different discretization should give different results
        samples10 = simple_curve.sample(10)
        assert samples10.shape == (10, 3)
        assert id(samples10) != id(samples)  # Different object

        # Verify that sample returns the same points as evaluate with linspace
        t_values = np.linspace(0, 1, 5)
        eval_points = simple_curve.evaluate(t_values)
        assert np.allclose(samples, eval_points)

    def test_length(self, simple_curve):
        length = simple_curve.length()
        # Length should be positive
        assert length > 0.0
        # For this specific curve, the length will be greater than the direct distance
        direct_distance = np.sqrt(3)  # distance from (0,0,0) to (1,1,1)
        assert length >= direct_distance

    def test_plot(self, simple_curve):
        # Create a 3D axis for plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Simply test that the plot method runs without errors
        simple_curve.plot(ax, "blue", 0.1)
        plt.close(fig)  # Close to avoid showing the plot during tests

    def test_bending_energy(self, simple_curve):
        energy = simple_curve._bending_energy(num_points=50)
        # Energy should be positive
        assert energy >= 0.0

        # Test with different number of points
        energy2 = simple_curve._bending_energy(num_points=100)
        # Should be roughly the same (might be slightly different due to discretization)
        assert np.isclose(energy, energy2, rtol=0.1)

    def test_twisting_energy(self, simple_curve):
        energy = simple_curve._twisting_energy(num_points=50)
        # Energy should be positive
        assert energy >= 0.0

        # Test with different number of points
        energy2 = simple_curve._twisting_energy(num_points=100)
        # Should be roughly the same (might be slightly different due to discretization)
        assert np.isclose(energy, energy2, rtol=0.1)

    def test_total_energy(self, simple_curve):
        energy = simple_curve.energy(num_points=50, alpha=0.8)
        # Energy should be positive
        assert energy >= 0.0

        # Test with different alpha
        energy2 = simple_curve.energy(num_points=50, alpha=0.5)
        # Should be different since we changed alpha
        assert energy != energy2

        # Energy should be the weighted sum of bending and twisting energies
        bending = simple_curve._bending_energy(num_points=50)
        twisting = simple_curve._twisting_energy(num_points=50)
        expected_energy = 0.5 * bending + 0.5 * 0.8 * twisting
        assert np.isclose(energy, expected_energy)
