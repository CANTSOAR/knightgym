"""Discrete action encoding shared by the Gymnasium and PettingZoo envs."""
from __future__ import annotations

import numpy as np

# kind: 0 noop, 1 limit, 2 market, 3 cancel all orders in `asset`
N_KINDS = 4
# limit price = reference (last valid mid, cents) + OFFSETS[i], in cents
OFFSETS = np.array([-100, -60, -35, -20, -12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100])
SIZES = np.array([1, 2, 3, 5, 10, 20])


def action_nvec(n_assets: int) -> list[int]:
    return [N_KINDS, n_assets, 2, len(OFFSETS), len(SIZES)]


def decode(market, agent: int, a) -> list[tuple]:
    """MultiDiscrete [kind, asset, side, offset_idx, size_idx] -> raw market action tuples."""
    kind, asset, side, oi, si = (int(x) for x in a)
    if kind == 0:
        return []
    if kind == 1:
        price = market.ref_cents(asset) + int(OFFSETS[oi])
        return [(agent, 1, asset, side, max(price, 1), int(SIZES[si]))]
    if kind == 2:
        return [(agent, 2, asset, side, 0, int(SIZES[si]))]
    return [(agent, 3, asset, 0, 0, 0)]


def encode_limit(side: int, asset: int, offset_cents: int, size: int, n_assets: int = 1):
    """Nearest discrete action to a requested (offset, size)."""
    oi = int(np.abs(OFFSETS - offset_cents).argmin())
    si = int(np.abs(SIZES - size).argmin())
    return np.array([1, asset, side, oi, si])
