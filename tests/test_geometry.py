"""Bearing sign convention: 0 = ahead, POSITIVE = LEFT (Doom ANGLE is counter-clockwise).
Prior art got this wrong and the bot spun in place."""
import pytest

from s1doom.perception.geometry import aim_tol, bearing_deg, norm_deg, range_bucket, side_of, turn_command


@pytest.mark.parametrize("angle", [0, 45, 90, 180, 270, 359])
def test_monster_on_left_has_positive_bearing(angle):
    import math
    a = math.radians(angle + 30)  # 30 deg counter-clockwise of facing = to the LEFT
    b = bearing_deg(0, 0, angle, 100 * math.cos(a), 100 * math.sin(a))
    assert b == pytest.approx(30, abs=1e-6)
    assert side_of(b) == "left"


@pytest.mark.parametrize("angle", [0, 90, 200])
def test_monster_on_right_has_negative_bearing(angle):
    import math
    a = math.radians(angle - 50)
    b = bearing_deg(0, 0, angle, 100 * math.cos(a), 100 * math.sin(a))
    assert b == pytest.approx(-50, abs=1e-6)
    assert side_of(b) == "right"


def test_facing_east_north_is_left():
    assert bearing_deg(0, 0, 0, 0, 100) == pytest.approx(90)
    assert bearing_deg(0, 0, 0, 0, -100) == pytest.approx(-90)


def test_turn_command_reduces_bearing():
    # ViZDoom TURN_LEFT_RIGHT_DELTA > 0 turns RIGHT (verified on 1.3.1), so a target on
    # the left (positive bearing) needs a NEGATIVE command.
    assert turn_command(30) < 0
    assert turn_command(-30) > 0
    assert abs(turn_command(90)) <= 12


def test_norm_and_buckets():
    assert norm_deg(190) == pytest.approx(-170)
    assert norm_deg(-180) == pytest.approx(180)
    assert range_bucket(50) == "melee" and range_bucket(200) == "close"
    assert range_bucket(500) == "mid" and range_bucket(2000) == "far"
    assert aim_tol(2000) >= 1.5 and aim_tol(64) > aim_tol(512)
