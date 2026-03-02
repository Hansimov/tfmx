"""Tests for tfmx.gpus.mon module (pure functions for fan curve logic)"""

import pytest

from tfmx.gpus.mon import (
    round_speed_up,
    parse_curve_points,
    curve_points_to_str,
    curve_points_to_display,
    calculate_fan_speed,
)


class TestRoundSpeedUp:
    """Test round_speed_up function"""

    def test_zero(self):
        assert round_speed_up(0) == 0

    def test_one(self):
        assert round_speed_up(1) == 5

    def test_exact_step(self):
        assert round_speed_up(5) == 5

    def test_round_up(self):
        assert round_speed_up(6) == 10

    def test_77(self):
        assert round_speed_up(77) == 80

    def test_100(self):
        assert round_speed_up(100) == 100

    def test_over_100(self):
        assert round_speed_up(105) == 100

    def test_negative(self):
        assert round_speed_up(-5) == 0

    def test_custom_step(self):
        assert round_speed_up(7, step=10) == 10
        assert round_speed_up(11, step=10) == 20


class TestParseCurvePoints:
    """Test parse_curve_points function"""

    def test_basic_curve(self):
        points = parse_curve_points("50-80/75-100")
        assert points == [(50, 80), (75, 100)]

    def test_single_point(self):
        points = parse_curve_points("30-50")
        assert points == [(30, 50)]

    def test_multiple_points(self):
        points = parse_curve_points("30-50/50-65/60-80/75-100")
        assert len(points) == 4
        assert points[0] == (30, 50)
        assert points[-1] == (75, 100)

    def test_unsorted_gets_sorted(self):
        points = parse_curve_points("75-100/50-80")
        assert points == [(50, 80), (75, 100)]

    def test_none_input(self):
        assert parse_curve_points(None) is None

    def test_empty_input(self):
        assert parse_curve_points("") is None

    def test_whitespace_input(self):
        assert parse_curve_points("   ") is None

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_curve_points("invalid")

    def test_invalid_temp_range(self):
        with pytest.raises(ValueError):
            parse_curve_points("130-50")  # temp > 120

    def test_invalid_speed_range(self):
        with pytest.raises(ValueError):
            parse_curve_points("50-110")  # speed > 100


class TestCurvePointsToStr:
    def test_basic(self):
        assert curve_points_to_str([(50, 80), (75, 100)]) == "50-80/75-100"

    def test_empty(self):
        assert curve_points_to_str([]) == "auto"


class TestCurvePointsToDisplay:
    def test_basic(self):
        result = curve_points_to_display([(50, 80), (75, 100)])
        assert result == "50°C->80%, 75°C->100%"

    def test_empty(self):
        assert curve_points_to_display([]) == "auto"


class TestCalculateFanSpeed:
    """Test fan speed calculation with linear interpolation"""

    def test_below_min(self):
        """Temperature below lowest point returns lowest speed"""
        speed = calculate_fan_speed(20, [(30, 50), (75, 100)])
        assert speed == 50

    def test_above_max(self):
        """Temperature above highest point returns highest speed"""
        speed = calculate_fan_speed(90, [(30, 50), (75, 100)])
        assert speed == 100

    def test_exact_point(self):
        """Temperature exactly at a point"""
        speed = calculate_fan_speed(30, [(30, 50), (75, 100)])
        assert speed == 50

    def test_interpolation_midpoint(self):
        """Linear interpolation between points"""
        # Midpoint between (30, 50) and (75, 100)
        # temp=52.5 => speed=75 (midpoint)
        speed = calculate_fan_speed(52, [(30, 50), (75, 100)])
        # (52 - 30) / (75 - 30) * (100 - 50) + 50 = 22/45 * 50 + 50 ≈ 74.4
        assert 74 <= speed <= 75

    def test_empty_curve(self):
        """No curve points returns None (auto mode)"""
        assert calculate_fan_speed(50, []) is None

    def test_single_point_below(self):
        speed = calculate_fan_speed(20, [(50, 80)])
        assert speed == 80

    def test_single_point_above(self):
        speed = calculate_fan_speed(60, [(50, 80)])
        assert speed == 80

    def test_three_points(self):
        """Three-point curve"""
        curve = [(30, 40), (50, 60), (70, 100)]
        assert calculate_fan_speed(25, curve) == 40  # below min
        assert calculate_fan_speed(50, curve) == 60  # exact mid point
        assert calculate_fan_speed(80, curve) == 100  # above max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
