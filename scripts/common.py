"""Shared evaluation helpers for baselines and plots."""
import numpy as np

from knightgym import KnightGymEnv, base_config
from knightgym.agents import ScriptedMarketMaker

EVAL_SEEDS = list(range(10_000, 10_050))


def make_env(**kw):
    return KnightGymEnv(base_config(**kw))


def run_episode(env, act, seed):
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, te, tr, info = env.step(act(obs, env))
        total += r
        done = te or tr
    return total, info["inventory"][0]


def evaluate(act, seeds=EVAL_SEEDS, **kw):
    env = make_env(**kw)
    env.action_space.seed(0)  # makes the random-policy baseline reproducible
    res = [run_episode(env, act, s) for s in seeds]
    pnl = np.array([r[0] for r in res])
    return pnl, np.array([r[1] for r in res])


def random_policy(obs, env):
    return env.action_space.sample()


class MMPolicy:
    """The scripted market maker run through the same discrete action interface:
    cancel+requote is two raw actions, so it drives the market directly."""
    def __init__(self, **kw):
        self.mm = ScriptedMarketMaker(**kw)


def evaluate_mm(seeds=EVAL_SEEDS, **mm_kw):
    """Scripted MM controls agent 0 directly through the raw action interface."""
    cfg = base_config()
    env = KnightGymEnv(cfg)
    pnl, inv = [], []
    for s in seeds:
        env.reset(seed=s)
        mm = ScriptedMarketMaker(**mm_kw)
        for _ in range(cfg["episode_len"]):
            env.market.step(mm(env.market, 0))
        pnl.append(env.market.equity(0, cfg["marking"]) - 10_000.0)
        inv.append(env.market.account(0)[1][0])
    return np.array(pnl), np.array(inv)
