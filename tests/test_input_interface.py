import sys
import os
import pytest
import tempfile
import json
import os
from unittest.mock import patch

currentdir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.abspath(os.path.join(currentdir, "../"))
sys.path.append(target_dir)

from input_interface import input_interface

@pytest.mark.parametrize("input_sequence, expected_group_count, output_filename", [
    (
        [
            # Point 0: air input (not linkage)
            "0,0,0", "0,0,1", "false","no",

            # Point 1 & 2: linkage pair on Z-plane, 1.7912 apart in Z
            "10,10,0", "1,0,0", "true","no",
            "10,10,1.7912", "1,0,0", "true","no",

            # Point 3 & 4: linkage pair on Y-plane
            "20,0,10", "0,0,1", "true","no",
            "20,1.7912,10", "0,0,1", "true","no",

            # Point 5 & 6: linkage pair on X-plane
            "0,20,20", "0,0,1", "true","no",
            "1.7912,20,20", "0,0,1", "true","no",

            # Point 7 & 8: linkage pair on XZ-plane
            "30,0,0", "0,-1,0", "true","no",
            "30,0,1.7912", "0,-1,0", "true","no",

            # Point 9: second air input
            "50,50,50", "-1,0,0", "false","yes",

            # Group connections: air inputs connect to 4 linkage points each
            "2,4,6,8",
            "2,3,4,5",
            "0.15"
        ],
        2,
        f"{target_dir}/paired_points.json"
    )
])

def test_input_interface_with_save(monkeypatch, input_sequence, expected_group_count, output_filename):
    input_iter = iter(input_sequence)
    monkeypatch.setattr("builtins.input", lambda _: next(input_iter))

    SAVE_TO_REAL_FILE = True
    if SAVE_TO_REAL_FILE:
        output_path = output_filename
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        output_path = os.path.join(tempfile.gettempdir(), "test.json")

    monkeypatch.setattr("input_interface.json_path", output_path)

    input_interface()

    with open(output_path) as f:
        data = json.load(f)

    assert "pipe_radius" in data
    assert isinstance(data["pipe_radius"], float)
    assert len(data["connections"]) == expected_group_count
