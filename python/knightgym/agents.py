"""Scripted opponents / baselines. Each is a callable (market, agent_id) -> raw action tuples
(agent, kind, asset, side, price_cents, qty). Written from scratch; they only use the public
market views (book, own account), never hidden fair value."""
from __future__ import annotations

import numpy as np


class ScriptedMarketMaker:
    """Re-quotes both sides around the reference mid every step, skewing quotes against
    inventory and standing down at the position cap."""

    def __init__(self, half_spread=4, size=2, skew_cents_per_unit=0.15, max_inv=40, asset=0):
        self.hs, self.size, self.skew, self.max_inv, self.asset = half_spread, size, skew_cents_per_unit, max_inv, asset

    def __call__(self, market, agent):
        k = self.asset
        ref = market.ref_cents(k)
        inv = market.account(agent)[1][k]
        shift = int(round(self.skew * inv))
        acts = [(agent, 3, k, 0, 0, 0)]
        if inv < self.max_inv:
            acts.append((agent, 1, k, 0, max(ref - self.hs - shift, 1), self.size))
        if inv > -self.max_inv:
            acts.append((agent, 1, k, 1, ref + self.hs - shift, self.size))
        return acts


class MomentumTaker:
    """Market-buys after `n` consecutive up-moves of the reference price, sells after `n` down-moves."""

    def __init__(self, n=3, size=1, asset=0):
        self.n, self.size, self.asset = n, size, asset
        self.hist: list[int] = []

    def reset(self):
        self.hist = []

    def __call__(self, market, agent):
        self.hist.append(market.ref_cents(self.asset))
        h = self.hist[-(self.n + 1):]
        if len(h) <= self.n:
            return []
        d = np.diff(h)
        if (d > 0).all():
            return [(agent, 2, self.asset, 0, 0, self.size)]
        if (d < 0).all():
            return [(agent, 2, self.asset, 1, 0, self.size)]
        return []


class MeanReverter:
    """Fades moves of the reference price away from its exponential moving average."""

    def __init__(self, alpha=0.1, thresh=6.0, size=1, asset=0):
        self.alpha, self.thresh, self.size, self.asset = alpha, thresh, size, asset
        self.ema = None

    def reset(self):
        self.ema = None

    def __call__(self, market, agent):
        r = market.ref_cents(self.asset)
        self.ema = r if self.ema is None else (1 - self.alpha) * self.ema + self.alpha * r
        dev = r - self.ema
        if dev > self.thresh:
            return [(agent, 2, self.asset, 1, 0, self.size)]
        if dev < -self.thresh:
            return [(agent, 2, self.asset, 0, 0, self.size)]
        return []
