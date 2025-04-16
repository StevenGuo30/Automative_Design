import pytest
import numpy as np
import matplotlib.pyplot as plt
from geometry.line import Line


class TestLine:
    @pytest.fixture
    def simple_line(self):
        # Create a line from (0,0,0) to (1,1,1)
        return Line(start=np.array([0.0, 0.0, 0.0]), end=np.array([1.0, 1.0, 1.0]))

    def test_initialization(self, simple_line):
        assert np.array_equal(simple_line.start, np.array([0.0, 0.0, 0.0]))
        assert np.array_equal(simple_line.end, np.array([1.0, 1.0, 1.0]))

    def test_evaluate_single_point(self, simple_line):
        # Middle point should be (0.5, 0.5, 0.5)
        result = simple_line.evaluate(0.5)
        assert np.allclose(result, np.array([0.5, 0.5, 0.5]))

        # Start point
        result = simple_line.evaluate(0.0)
        assert np.allclose(result, np.array([0.0, 0.0, 0.0]))

        # End point
        result = simple_line.evaluate(1.0)
        assert np.allclose(result, np.array([1.0, 1.0, 1.0]))

    def test_evaluate_multiple_points(self, simple_line):
        t_values = np.array([0.0, 0.5, 1.0])
        expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        result = simple_line.evaluate(t_values)
        assert np.allclose(result, expected)

    def test_sample(self, simple_line):
        # Test with discretization of 5 points
        samples = simple_line.sample(5)
        assert samples.shape == (5, 3)  # 5 points, each with 3 coordinates

        # First point should be start point
        assert np.allclose(samples[0], simple_line.start)

        # Last point should be end point
        assert np.allclose(samples[-1], simple_line.end)

        # Middle points should be evenly spaced
        expected_samples = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
                [0.75, 0.75, 0.75],
                [1.0, 1.0, 1.0],
            ]
        )
        assert np.allclose(samples, expected_samples)

        # Test caching behavior
        # Second call should return the cached result
        cached_samples = simple_line.sample(5)
        assert id(cached_samples) == id(samples)  # Should return the exact same object

        # Different discretization should give different results
        samples10 = simple_line.sample(10)
        assert samples10.shape == (10, 3)
        assert id(samples10) != id(samples)  # Different object

    def test_energy(self, simple_line):
        energy = simple_line.energy()
        assert energy == 0.0  # Line energy should be zero

    def test_length(self, simple_line):
        length = simple_line.length()
        # Length of this line should be sqrt(3)
        expected_length = np.sqrt(3)
        assert np.isclose(length, expected_length)

        # Test another line for verification
        line2 = Line(start=np.array([0.0, 0.0, 0.0]), end=np.array([3.0, 4.0, 0.0]))
        # Length should be 5.0 (3-4-5 triangle)
        assert np.isclose(line2.length(), 5.0)

    def test_plot(self, simple_line):
        # Create a 3D axis for plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Simply test that the plot method runs without errors
        # We can't easily check the resulting plot automatically
        simple_line.plot(ax, "blue", 0.1)
        plt.close(fig)  # Close to avoid showing the plot during tests
