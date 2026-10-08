"""Default scenario configs. Prices are dollars, sizes are units, spreads are cents."""
from __future__ import annotations

import copy
import json

LIQUID = dict(name="LIQ", p0=100.0, vol_cents=4.0, house_levels=3, house_size=10.0,
              half_spread_cents=5, level_gap_cents=3, noise_rate=0.6, noise_size=3)

# One house level, small size, wide spread, rare noise flow: easy to push around.
THIN = dict(name="THIN", p0=20.0, vol_cents=3.0, house_levels=1, house_size=2.0,
            half_spread_cents=25, level_gap_cents=10, noise_rate=0.08, noise_size=1)


def base_config(**overrides) -> dict:
    cfg = dict(n_agents=1, episode_len=200, marking="mid", assets=[dict(LIQUID)])
    cfg.update(copy.deepcopy(overrides))
    return cfg


def to_json(cfg: dict) -> str:
    return json.dumps(cfg)
