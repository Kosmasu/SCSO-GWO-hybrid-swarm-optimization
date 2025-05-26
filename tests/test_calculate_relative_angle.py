import pytest
import math
from game import calculate_relative_angle

@pytest.mark.parametrize("angle, x1, y1, x2, y2, expected", [
    # Test case 1: Ship facing right (0 radians), target directly to the right
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    
    # Test case 2: Ship facing right (0 radians), target directly above
    (0.0, 0.0, 0.0, 0.0, 1.0, math.pi/2),
    
    # Test case 3: Ship facing right (0 radians), target directly below
    (0.0, 0.0, 0.0, 0.0, -1.0, -math.pi/2),
    
    # Test case 4: Ship facing right (0 radians), target directly to the left
    (0.0, 0.0, 0.0, -1.0, 0.0, math.pi),
    
    # Test case 5: Ship facing up (π/2), target directly up
    (math.pi/2, 0.0, 0.0, 0.0, 1.0, 0.0),
    
    # Test case 6: Ship facing up (π/2), target directly right
    (math.pi/2, 0.0, 0.0, 1.0, 0.0, -math.pi/2),
    
    # Test case 7: Ship facing down (-π/2), target directly down
    (-math.pi/2, 0.0, 0.0, 0.0, -1.0, 0.0),
    
    # Test case 8: Ship facing left (π), target directly left
    (math.pi, 0.0, 0.0, -1.0, 0.0, 0.0),
    
    # Test case 9: 45-degree angle test
    (0.0, 0.0, 0.0, 1.0, 1.0, math.pi/4),
    
    # Test case 10: Wrap around test - large positive angle
    (math.pi/4, 0.0, 0.0, -1.0, -1.0, 3*math.pi/4),
    
    # Test case 11: Same position (no relative angle)
    (0.0, 5.0, 5.0, 5.0, 5.0, 0.0),
    
    # Test case 12: Full circle test - should wrap to negative
    (0.1, 0.0, 0.0, -1.0, 0.0, math.pi - 0.1),
])
def test_calculate_relative_angle_parameterized(angle, x1, y1, x2, y2, expected):
    result = calculate_relative_angle(angle, x1, y1, x2, y2)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

def test_result_in_valid_range():
    """Test that the result is always in the range [-π, π]"""
    test_cases = [
        (0.0, 0.0, 0.0, 1.0, 1.0),
        (math.pi, 0.0, 0.0, 1.0, 1.0),
        (2*math.pi, 0.0, 0.0, 1.0, 1.0),
        (-math.pi, 0.0, 0.0, 1.0, 1.0),
        (3*math.pi, 0.0, 0.0, 1.0, 1.0),
    ]
    
    for angle, x1, y1, x2, y2 in test_cases:
        result = calculate_relative_angle(angle, x1, y1, x2, y2)
        assert -math.pi <= result <= math.pi, f"Result {result} out of range [-π, π]"

def test_edge_case_zero_distance():
    """Test when ship and target are at the same position"""
    result = calculate_relative_angle(0.0, 0.0, 0.0, 0.0, 0.0)
    assert result == 0.0

def test_negative_coordinates():
    """Test with negative coordinates"""
    result = calculate_relative_angle(0.0, -5.0, -3.0, -2.0, -1.0)
    expected = math.atan2(2.0, 3.0)  # dy=2, dx=3
    assert abs(result - expected) < 1e-10