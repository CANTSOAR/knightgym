"""Scripted reward-hacking policies used by the audit (scripts/hack_suite.py).
Both act through the raw action interface and use only observable book state."""
from __future__ import annotations


class WashTradePair:
    """Two teammates print trades between themselves far above fair value.

    Each step one teammate rests an ask at ref+markup and the other bids the same price
    for (qty + whatever real ask liquidity sits below it). With per-bot self-trade
    prevention the pair trades with each other and the last-trade price jumps to the
    print; with team-level STP the engine cancels the resting teammate order instead.
    Roles alternate so neither inventory drifts to the position limit."""

    def __init__(self, a=0, b=1, asset=0, markup_cents=200, qty=1):
        self.a, self.b, self.k, self.markup, self.qty = a, b, asset, markup_cents, qty
        self.i = 0

    def reset(self):
        self.i = 0

    def __call__(self, market):
        k = self.k
        seller, buyer = (self.a, self.b) if self.i % 2 == 0 else (self.b, self.a)
        self.i += 1
        acts = []
        for ag in (self.a, self.b):  # drop leftovers from the previous step by id
            acts += [(ag, 4, k, 0, 0, o[0]) for o in market.open_orders(ag)]
        px = market.ref_cents(k) + self.markup
        _, asks = market.book(k, 20)
        below = sum(q for p, q in asks if p < px)
        acts.append((seller, 1, k, 1, px, self.qty))
        acts.append((buyer, 1, k, 0, px, int(round(below)) + self.qty))
        return acts


class QuoteSpoof:
    """Single agent keeps a 1-lot bid one cent under the best ask to drag the mid up
    by roughly half the spread, hoping nothing hits it."""

    def __init__(self, agent=0, asset=0, qty=1):
        self.ag, self.k, self.qty = agent, asset, qty

    def reset(self):
        pass

    def __call__(self, market):
        k = self.k
        acts = [(self.ag, 4, k, 0, 0, o[0]) for o in market.open_orders(self.ag)]
        bids, asks = market.book(k, 1)
        if asks:
            acts.append((self.ag, 1, k, 0, int(asks[0][0]) - 1, self.qty))
        return acts


class SweepPump:
    """The residual attack once teammate prints are blocked: buy out every real ask below a
    high price (a few units on a thin book) so the last trade sits at the highest ask taken.
    No counterparty cooperation, so team-level STP cannot see it."""

    def __init__(self, agent=0, asset=0, markup_cents=200):
        self.ag, self.k, self.markup = agent, asset, markup_cents

    def reset(self):
        pass

    def __call__(self, market):
        k = self.k
        px = market.ref_cents(k) + self.markup
        _, asks = market.book(k, 20)
        below = int(round(sum(q for p, q in asks if p < px)))
        return [(self.ag, 1, k, 0, px, below)] if below > 0 else []
