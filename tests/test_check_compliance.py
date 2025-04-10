import pytest
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.abspath(os.path.join(currentdir, "../"))
sys.path.append(target_dir)

from input_interface import check_compliance

def make_point(name, coord, direction, is_link=True):
    return {
        "name": name,
        "coordinates": dict(zip(["x", "y", "z"], coord)),
        "directions": dict(zip(["x", "y", "z"], direction)),
        "is_linkage": is_link,
    }

def test_compliance_with_four_planes():
    # Construct 8 linkage points (in 4 planes, no interference)
    points = [
        make_point("A", [10, 10, 0], [1, 0, 0]),
        make_point("B", [10, 10, 1.7912], [1, 0, 0]),
        make_point("C", [20, 0, 10], [0, 0, 1]),
        make_point("D", [20, 1.7912, 10], [0, 0, 1]),
        make_point("E", [0, 20, 20], [0, 0, 1]),
        make_point("F", [1.7912, 20, 20], [0, 0, 1]),
        make_point("G", [30, 0, 0], [0, -1, 0]),
        make_point("H", [30, 0, 1.7912], [0, -1, 0]),
    ]
    result = check_compliance(points)
    assert result == []