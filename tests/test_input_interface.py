import sys
import os
import pytest

currentdir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.abspath(os.path.join(currentdir, "../"))
sys.path.append(target_dir)

from input_interface import input_interface

@pytest.mark.parametrize("points_input, expected_points", [
    (["0, 0, 0","0,0,1",'False', 'no', # First test case is a branch example
      "1, 1, 0","1,1,0",'True', 'no',
      "2, 0, 1","0,0,1", 'True', 'yes',
      '2,3',# 2,3 is the pair points
      '0.01'], None), # 0.01 is the pipe radius
    (["0, 0, 0","0,0,1", 'False', 'no', # Second test case is a switch example
      "0, 1, 0","0,0,1", 'False', 'no',
      "2, 0, 1","0,1,0", 'True', 'no',
      "3, 1, 0","0,1,0", 'True', 'no',
      "2, 2, 0","0,0,1", 'True', 'yes',
      '3,5', # 1,3 5is connected
      '2',# 2,4 is connected
      '0.01'], None), # 0.01 is the pipe radius
    (
      [
          # Group 1: A simple straight pipe with 3 points
          "0, 0, 0", "1, 0, 0", "False", "no",  # Point 1
          "1, 0, 0", "1, 0, 0", "False", "no",  # Point 2
          "2, 0, 0", "1, 0, 0", "True", "no",   # Point 3

          # Group 2: A T-junction with 4 points
          "1, 1, 0", "0, 1, 0", "False", "no",  # Point 4
          "1, 2, 0", "0, 1, 0", "True", "no",   # Point 5
          "0, 1, 0", "1, 0, 0", "True", "no",  # Point 6
          "2, 1, 0", "1, 0, 0", "True", "no",   # Point 7

          # Group 3: A corner with 4 points
          "3, 0, 0", "1, 0, 0", "False", "no",  # Point 8
          "4, 0, 0", "1, 0, 0", "False", "no",  # Point 9
          "4, 1, 0", "0, 1, 0", "False", "no",  # Point 10
          "4, 2, 0", "0, 1, 0", "True", "no",   # Point 11

          # Group 4: A diagonal pair
          "5, 5, 0", "1, 1, 0", "False", "no",  # Point 12
          "6, 6, 0", "1, 1, 0", "True", "yes",  # Point 13

          # Pair connections (each line matches prompt after input collection)
          "12,13",  # Connect Point 1 with 12 and 13
          "8,9,10",  # Connect Point 2 with 9,10,11
          "4,5,6",  # Connect Point 3 with 6,7,8
          "2",  # Connect Point 4 with 5

          # Pipe radius
          "0.05"
      ],
      None
  ),
])

 
def test_input_interface(monkeypatch,capfd, points_input, expected_points):
    # Test input_interface
    # mimic user input
    input_iter = iter(points_input)
    monkeypatch.setattr('builtins.input', lambda _: next(input_iter)) 
    
    # Capture stdout and stderr
    out,err=capfd.readouterr()
    print(out)
    print(err)
    
    assert input_interface() == expected_points