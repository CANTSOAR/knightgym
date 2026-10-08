"""PettingZoo ParallelEnv: every agent is externally controlled; teams only matter for
self-trade prevention (config["team_stp"]) and for team-level accounting in info."""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from . import _rs
from .actions import action_nvec, decode
from .config import base_config


class KnightGymParallel(ParallelEnv):
    metadata = {"name": "knightgym_parallel_v0"}

    def __init__(self, config: Optional[dict] = None):
        self.config = config or base_config(n_agents=2)
        self.market = _rs.Market(json.dumps(self.config))
        n = self.market.n_agents
        self.possible_agents = [f"agent_{i}" for i in range(n)]
        self.agents = list(self.possible_agents)
        self.team_of = self.config.get("team_of") or list(range(n))
        self._obs_space = spaces.Box(-np.inf, np.inf, (self.market.obs_dim(),), np.float32)
        self._act_space = spaces.MultiDiscrete(action_nvec(self.market.n_assets))
        self._rng = np.random.default_rng()

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.market.reset(int(self._rng.integers(0, 2**31 - 1)))
        self.agents = list(self.possible_agents)
        return self._obs(), {a: self._info(i, []) for i, a in enumerate(self.agents)}

    def _obs(self):
        return {a: self.market.obs(i) for i, a in enumerate(self.possible_agents)}

    def _info(self, i, fills):
        cash, inv = self.market.account(i)
        return dict(cash=cash, inventory=inv, team=self.team_of[i],
                    equity={r: self.market.equity(i, r) for r in ("last", "mid", "vwap")},
                    fills=[f for f in fills if f[4] == i or f[5] == i])

    def step(self, actions: dict):
        raw = []
        for i, a in enumerate(self.possible_agents):
            if a in actions:
                raw += decode(self.market, i, actions[a])
        out = self.market.step(raw)
        trunc = bool(out["done"])
        obs = self._obs()
        rew = {a: float(out["rewards"][i]) for i, a in enumerate(self.possible_agents)}
        term = {a: False for a in self.possible_agents}
        truncs = {a: trunc for a in self.possible_agents}
        infos = {a: self._info(i, out["fills"]) for i, a in enumerate(self.possible_agents)}
        if trunc:
            self.agents = []
        return obs, rew, term, truncs, infos
