import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import knightgym
from knightgym import KnightGymEnv, KnightGymParallel, base_config
from knightgym.agents import ScriptedMarketMaker
from knightgym.actions import decode


def rollout(seed, cfg=None, steps=60, opponents=None):
    env = KnightGymEnv(cfg or base_config(), opponents=opponents)
    obs, _ = env.reset(seed=seed)
    trace = [obs.copy()]
    rng = np.random.default_rng(123)  # action stream independent of env seed
    for _ in range(steps):
        a = np.array([rng.integers(n) for n in env.action_space.nvec])
        obs, r, te, tr, info = env.step(a)
        trace.append(obs.copy())
        trace.append(np.array([r, info["equity"]["last"], info["equity"]["mid"]]))
    return np.concatenate(trace)


def test_gymnasium_api_check():
    env = KnightGymEnv(base_config())
    check_env(env, skip_render_check=True)


def test_registered_and_three_line_example():
    env = gym.make("KnightGym-v0")
    obs, info = env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape and np.isfinite(r)


def test_determinism_same_seed_same_trajectory():
    assert np.array_equal(rollout(7), rollout(7))
    assert not np.array_equal(rollout(7), rollout(8))


def test_determinism_with_opponents():
    mk = lambda: {1: ScriptedMarketMaker()}
    cfg = base_config(n_agents=2)
    assert np.array_equal(rollout(3, cfg, opponents=mk()), rollout(3, cfg, opponents=mk()))


def test_time_only_advances_on_step():
    env = KnightGymEnv(base_config())
    env.reset(seed=1)
    m = env.market
    before = (m.t, m.book(0, 5), m.account(0))
    for _ in range(10):  # reading state must not move the clock or the book
        env.market.obs(0); env.market.marks("mid"); env.market.equity(0, "last")
    assert (m.t, m.book(0, 5), m.account(0)) == before
    env.step(env.action_space.sample())
    assert m.t == before[0] + 1


def test_episode_truncates_at_horizon():
    env = KnightGymEnv(base_config(episode_len=15))
    env.reset(seed=0)
    for i in range(15):
        _, _, te, tr, _ = env.step(np.zeros(5, dtype=int))
    assert tr and not te


def test_limit_order_rests_and_cancel_all_removes_it():
    env = KnightGymEnv(base_config())
    env.reset(seed=0)
    m = env.market
    ref = m.ref_cents(0)
    m.step([(0, 1, 0, 0, ref - 50, 2)])  # deep passive bid
    assert any(o[3] == ref - 50 for o in m.open_orders(0))
    m.step([(0, 3, 0, 0, 0, 0)])
    assert m.open_orders(0) == []


def test_market_buy_fills_and_pnl_consistent():
    env = KnightGymEnv(base_config(marking="last", assets=[dict(knightgym.LIQUID, noise_rate=0.0, vol_cents=0.0)]))
    env.reset(seed=0)
    m = env.market
    cash0, _ = m.account(0)
    out = m.step([(0, 2, 0, 0, 0, 3)])
    mine = [f for f in out["fills"] if f[4] == 0]
    assert sum(f[3] for f in mine) == 3
    cash1, inv = m.account(0)
    assert inv[0] == 3
    assert cash0 - cash1 == pytest.approx(sum(f[2] * f[3] for f in mine))
    # reward == equity change under the configured rule
    assert out["rewards"][0] == pytest.approx(m.equity(0, "last") - 10_000.0)


def test_marking_rules_differ_on_same_state():
    env = KnightGymEnv(base_config())
    env.reset(seed=0)
    for _ in range(30):
        env.step(env.action_space.sample())
    marks = [env.market.marks(r)[0] for r in ("last", "mid", "vwap")]
    assert all(m > 0 for m in marks) and len(set(marks)) > 1
    with pytest.raises(ValueError):
        env.market.marks("bogus")


def test_position_limit_rejects():
    env = KnightGymEnv(base_config(pos_limit=5))
    env.reset(seed=0)
    out = env.market.step([(0, 2, 0, 0, 0, 20)])
    assert out["rejects"] == [0]


def test_teammate_cross_trades_only_without_team_stp():
    thin = dict(knightgym.THIN, house_levels=0)
    for team_stp in (False, True):
        cfg = base_config(n_agents=2, team_of=[0, 0], team_stp=team_stp, assets=[thin])
        env = KnightGymEnv(cfg)
        env.reset(seed=0)
        m = env.market
        m.step([(0, 1, 0, 1, 2500, 2)])
        out = m.step([(1, 1, 0, 0, 2500, 2)])
        wash = [f for f in out["fills"] if f[4] == 1 and f[5] == 0]
        assert bool(wash) == (not team_stp)
        if team_stp:
            assert out["stp_cancels"], "resting teammate order should be STP-cancelled"


def test_pettingzoo_parallel_api():
    from pettingzoo.test import parallel_api_test
    env = KnightGymParallel(base_config(n_agents=3, team_of=[0, 0, 1]))
    parallel_api_test(env, num_cycles=50)


def test_pettingzoo_determinism():
    def run():
        env = KnightGymParallel(base_config(n_agents=2))
        env.reset(seed=5)
        rng = np.random.default_rng(0)
        tot = []
        for _ in range(40):
            acts = {a: np.array([rng.integers(n) for n in env.action_space(a).nvec]) for a in env.agents}
            _, r, *_ = env.step(acts)
            tot.append(tuple(r.values()))
        return tot
    assert run() == run()


def test_determinism_across_processes():
    import subprocess, sys
    code = ("import numpy as np; from knightgym import KnightGymEnv, base_config\n"
            "e=KnightGymEnv(base_config()); o,_=e.reset(seed=11); r=np.random.default_rng(0); s=0.0\n"
            "for _ in range(80):\n"
            "    o,rw,*_=e.step(np.array([r.integers(n) for n in e.action_space.nvec])); s+=rw+o.sum()\n"
            "print(repr(float(s)))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout for _ in range(3)}
    assert len(outs) == 1
