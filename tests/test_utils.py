import pytest
import numpy as np
from geometry.utils import nearest_distance_between_geometry
from geometry.line import Line


class TestUtils:
    @pytest.fixture
    def parallel_lines(self):
        # Create two parallel lines with distance 1.0 between them
        line1 = Line(start=np.array([0.0, 0.0, 0.0]), end=np.array([1.0, 0.0, 0.0]))
        line2 = Line(start=np.array([0.0, 1.0, 0.0]), end=np.array([1.0, 1.0, 0.0]))
        return line1, line2

    @pytest.fixture
    def perpendicular_lines(self):
        # Create two perpendicular lines that meet at (1,1,0)
        line1 = Line(start=np.array([0.0, 1.0, 0.0]), end=np.array([2.0, 1.0, 0.0]))
        line2 = Line(start=np.array([1.0, 0.0, 0.0]), end=np.array([1.0, 2.0, 0.0]))
        return line1, line2

    @pytest.fixture
    def skew_lines(self):
        # Create two skew lines that don't intersect
        line1 = Line(start=np.array([0.0, 0.0, 0.0]), end=np.array([1.0, 0.0, 0.0]))
        line2 = Line(start=np.array([0.0, 1.0, 1.0]), end=np.array([1.0, 1.0, 1.0]))
        return line1, line2

    def test_nearest_distance_parallel_lines(self, parallel_lines):
        line1, line2 = parallel_lines
        # Distance between these parallel lines should be 1.0
        distance = nearest_distance_between_geometry(line1, line2)
        assert np.isclose(distance, 1.0)

    def test_nearest_distance_perpendicular_lines(self, perpendicular_lines):
        line1, line2 = perpendicular_lines
        # Distance should be close to 0 since they intersect
        distance = nearest_distance_between_geometry(line1, line2)
        assert np.isclose(distance, 0.0, atol=1e-7)

    def test_nearest_distance_skew_lines(self, skew_lines):
        line1, line2 = skew_lines
        # Distance between these skew lines should be 1.414... (sqrt(2))
        distance = nearest_distance_between_geometry(line1, line2)
        assert np.isclose(distance, np.sqrt(2))

    def test_nearest_distance_discretization(self, parallel_lines):
        line1, line2 = parallel_lines

        # Test with default discretization
        distance1 = nearest_distance_between_geometry(line1, line2)

        # Test with higher discretization
        distance2 = nearest_distance_between_geometry(line1, line2, discretization=200)

        # Both should be close
        assert np.isclose(distance1, distance2)

        # Test with lower discretization
        distance3 = nearest_distance_between_geometry(line1, line2, discretization=10)

        # Should still be reasonably close
        assert np.isclose(distance1, distance3, rtol=0.1)
