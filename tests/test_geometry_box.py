import pytest
import numpy as np
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.abspath(os.path.join(currentdir, "../"))
sys.path.append(target_dir)

from Generate_path import is_geometry_in_box
from geometry.line import Line
from geometry.curve import Curve
from geometry.protocol import BOX


class MockGeometry:
    def __init__(self, points):
        self.points = np.array(points)

    def sample(self, discretization=100):
        return self.points


class TestIsGeometryInBox:
    @pytest.fixture
    def box_fixture(self):
        # Create a box from (-1, -1, -1) to (1, 1, 1)
        return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    @pytest.fixture
    def line_inside_box(self):
        # Line completely inside the box
        return Line(start=np.array([-0.5, -0.5, -0.5]), end=np.array([0.5, 0.5, 0.5]))

    @pytest.fixture
    def line_outside_box(self):
        # Line completely outside the box (in x direction)
        return Line(start=np.array([2.0, 0.0, 0.0]), end=np.array([3.0, 0.0, 0.0]))

    @pytest.fixture
    def line_crossing_box(self):
        # Line crossing the box boundary
        return Line(start=np.array([0.0, 0.0, 0.0]), end=np.array([2.0, 0.0, 0.0]))

    @pytest.fixture
    def simple_curve(self):
        # Hermite curve defined by two points and tangent directions
        return Curve(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([0.5, 0.5, 0.5]),
            direction_a=np.array([1.0, 0.0, 0.0]),
            direction_b=np.array([0.0, 1.0, 0.0]),
        )

    @pytest.fixture
    def mock_geometry(self):
        # Used to test specific sample points
        return MockGeometry

    def test_line_inside_box(self, line_inside_box, box_fixture):
        # Line inside box should return False (not outside)
        assert not is_geometry_in_box(line_inside_box, box_fixture)

    def test_line_outside_box(self, line_outside_box, box_fixture):
        # Line outside box should return True (is outside)
        assert is_geometry_in_box(line_outside_box, box_fixture)

    def test_line_crossing_box(self, line_crossing_box, box_fixture):
        # Line crossing box should return True (part is outside)
        assert is_geometry_in_box(line_crossing_box, box_fixture)

    def test_curve_inside_box(self, simple_curve, box_fixture):
        # Curve inside box should return False
        assert not is_geometry_in_box(simple_curve, box_fixture)

    def test_multiple_points(self, mock_geometry, box_fixture):
        # Test geometry with multiple points, some inside, some outside
        mixed_points = [
            [0.0, 0.0, 0.0],  # inside
            [0.5, 0.5, 0.5],  # inside
            [2.0, 0.0, 0.0],  # outside (x_max)
        ]
        geom_mixed = mock_geometry(mixed_points)
        assert is_geometry_in_box(geom_mixed, box_fixture)

        # All points inside
        inside_points = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]]
        geom_inside = mock_geometry(inside_points)
        assert not is_geometry_in_box(geom_inside, box_fixture)
