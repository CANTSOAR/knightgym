"""The reward-hack claims in the README, as regression tests (20 seeds each)."""
import sys
import numpy as np
sys.path.insert(0, "scripts")
import hack_suite as hs

SEEDS = range(500, 520)


def gain(attack, marking, team_stp, key="marked"):
    c = hs.cfg(marking, team_stp)
    d = [hs.episode(c, s, attack)[key] - hs.episode(c, s, None)[key] for s in SEEDS]
    return float(np.mean(d))


def test_exploit_exists_last_mark_bot_stp():
    assert gain("wash", "last", False) > 500           # reward inflated
    assert gain("wash", "last", False, "true") < 0     # while real wealth falls


def test_team_stp_removes_teammate_prints():
    c = hs.cfg("last", True)
    assert hs.episode(c, 500, "wash")["wash_fills"] == 0
    assert hs.episode(c, 500, "wash")["stp"] > 0


def test_mid_marking_tracks_true_wealth_under_wash():
    m, t = gain("wash", "mid", False), gain("wash", "mid", False, "true")
    assert abs(m - t) < 5


def test_patch_collapses_gain():
    before = gain("wash", "last", False)
    after = gain("wash", "mid", True)
    assert after < 0 < before and after < 0.05 * before
