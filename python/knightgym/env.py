"""Single-agent Gymnasium environment: one learner against house liquidity, noise
traders and optional scripted opponents sharing the same order books."""
from __future__ import annotations

import json
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import _rs
from .actions import action_nvec, decode
from .config import base_config

# An opponent is a callable (market, agent_id) -> list of raw action tuples.
Opponent = Callable[[object, int], list]


class KnightGymEnv(gym.Env):
    """Observation: per asset the top-`obs_depth` bid/ask levels (price relative to the
    reference mid, size), last trade, own inventory and resting size; plus cash/equity/time.
    Action: MultiDiscrete [kind, asset, side, price-offset, size]; see knightgym.actions.
    Reward: change in marked equity (dollars * reward_scale) under config["marking"].
    Time advances only in step(); all randomness derives from the reset seed."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[dict] = None, agent_id: int = 0,
                 opponents: Optional[dict[int, Opponent]] = None):
        super().__init__()
        self.config = config or base_config()
        self.agent_id = agent_id
        self.opponents = opponents or {}
        self.market = _rs.Market(json.dumps(self.config))
        self.n_assets = self.market.n_assets
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.market.obs_dim(),), np.float32)
        self.action_space = spaces.MultiDiscrete(action_nvec(self.n_assets))
        self._seeded = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        s = int(self.np_random.integers(0, 2**31 - 1))
        self.market.reset(s)
        for opp in self.opponents.values():
            getattr(opp, "reset", lambda: None)()
        return self.market.obs(self.agent_id), self._info([])

    def step(self, action):
        acts = decode(self.market, self.agent_id, action)
        for aid, opp in self.opponents.items():
            acts += opp(self.market, aid)
        out = self.market.step(acts)
        obs = self.market.obs(self.agent_id)
        reward = float(out["rewards"][self.agent_id])
        return obs, reward, False, bool(out["done"]), self._info(out["fills"], out)

    def _info(self, fills, out=None):
        a = self.agent_id
        cash, inv = self.market.account(a)
        info = dict(
            t=self.market.t, cash=cash, inventory=inv,
            equity={r: self.market.equity(a, r) for r in ("last", "mid", "vwap")},
            fills=[f for f in fills if f[4] == a or f[5] == a],
        )
        if out is not None:
            info["rejected"] = a in out["rejects"]
        return info
