import sys
import os
import pytest

currentdir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.abspath(os.path.join(currentdir, "../test_fusionapi_1"))
sys.path.append(target_dir)

from input_interface import input_interface

@pytest.mark.parametrize("points_input, expected_points", [
    (["0, 0, 0","0,0,1",'False', 'no', # First test case is a branch example
      "1, 1, 0","1,1,0",'True', 'no',
      "2, 0, 1","0,0,1", 'True', 'yes',
      '2,3'], None), # 2,3 is the pair points
    (["0, 0, 0","0,0,1", 'False', 'no', # Second test case is a switch example
      "0, 1, 0","0,0,1", 'False', 'no',
      "2, 0, 1","0,1,0", 'True', 'no',
      "3, 1, 0","0,1,0", 'True', 'yes',
      '3', # 1,3 is connected
      '2'], None), # 2,4 is connected
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