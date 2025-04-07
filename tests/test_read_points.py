import sys
import os
import pytest

currentdir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.abspath(os.path.join(currentdir, "../fusion_api_example/tempCodeRunnerFile/"))
sys.path.append(target_dir)

from tempCodeRunnerFile import read_points

def test_read_points():
    # Test read_points
    point_dict, group_connections = read_points()
    assert point_dict == {
    'A': [0.0, 0.0, 0.0],
    'C': [2.0, 0.0, 1.0],
    'E': [2.0, 2.0, 0.0],
    'B': [0.0, 1.0, 0.0],
    'D': [3.0, 1.0, 0.0]
    }
    assert group_connections == [['A', 'C', 'E'], ['B', 'D']]