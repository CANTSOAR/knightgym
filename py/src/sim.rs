//! Deterministic multi-agent market simulator on top of the exchange matching core.
//!
//! Time is a step counter that advances only in `Market::step`. One step:
//!   1. external agents' actions are applied (order shuffled by the seeded RNG),
//!   2. background noise traders send marketable orders,
//!   3. the latent fair value moves and the house market maker requotes
//!      (its fresh quotes can cross, and so pick off, stale agent orders),
//!   4. marks and rewards are computed.
//! Everything random flows from one ChaCha8 stream seeded in `reset(seed)`.
//!
//! Units: prices are integer core ticks (1e-4 $) internally and cents at the API;
//! quantity is integer lots (1e-3 unit) internally and units at the API.
//! Cash is stored in ticks*lots (1e-7 $) as i64.

use knightgym_core::matching::OrderBook;
use knightgym_core::models::{Order, OrderType, Side};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap, VecDeque};
use uuid::Uuid;

pub const TICKS_PER_CENT: i64 = 100;
pub const LOTS_PER_UNIT: i64 = 1000;
const CASH_SCALE: f64 = 1e7;
pub const HOUSE: i64 = -1;
pub const NOISE: i64 = -2;
const WARMUP_STEPS: u64 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MarkRule {
    Last,
    Mid,
    Vwap,
    /// Oracle: the latent fair value. Not observable by agents; used to measure true wealth.
    Fair,
}

impl MarkRule {
    pub fn index(self) -> usize {
        match self {
            MarkRule::Last => 0,
            MarkRule::Mid => 1,
            MarkRule::Vwap => 2,
            MarkRule::Fair => 3,
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "last" => Some(Self::Last),
            "mid" => Some(Self::Mid),
            "vwap" => Some(Self::Vwap),
            "fair" => Some(Self::Fair),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AssetCfg {
    pub name: String,
    /// Starting fair value, dollars.
    pub p0: f64,
    /// Std-dev of the fair-value random walk per step, cents.
    pub vol_cents: f64,
    pub house_levels: usize,
    /// Units quoted per house level per side.
    pub house_size: f64,
    pub half_spread_cents: i64,
    pub level_gap_cents: i64,
    /// Expected number of noise-trader market orders per step.
    pub noise_rate: f64,
    /// Noise order size is uniform in 1..=noise_size units.
    pub noise_size: i64,
}

impl Default for AssetCfg {
    fn default() -> Self {
        Self {
            name: "ABC".into(),
            p0: 100.0,
            vol_cents: 4.0,
            house_levels: 3,
            house_size: 10.0,
            half_spread_cents: 5,
            level_gap_cents: 3,
            noise_rate: 0.6,
            noise_size: 3,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub n_agents: usize,
    /// Team id per agent (defaults to one team per agent).
    pub team_of: Vec<i64>,
    /// If true, self-trade prevention keys on the team; otherwise on the individual bot.
    pub team_stp: bool,
    pub episode_len: u64,
    pub marking: MarkRule,
    pub vwap_window: u64,
    pub assets: Vec<AssetCfg>,
    pub init_cash: f64,
    /// init_inventory[agent][asset], units. Missing entries are zero.
    pub init_inventory: Vec<Vec<f64>>,
    pub pos_limit: f64,
    pub obs_depth: usize,
    pub reward_scale: f64,
    /// Per-step penalty coefficient on sum |inventory value| (0 disables).
    pub inv_penalty: f64,
    pub shuffle_actions: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_agents: 1,
            team_of: vec![],
            team_stp: false,
            episode_len: 200,
            marking: MarkRule::Mid,
            vwap_window: 20,
            assets: vec![AssetCfg::default()],
            init_cash: 10_000.0,
            init_inventory: vec![],
            pos_limit: 200.0,
            obs_depth: 3,
            reward_scale: 1.0,
            inv_penalty: 0.0,
            shuffle_actions: true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Action {
    Noop,
    Limit { asset: usize, side: Side, price_cents: i64, qty: i64 },
    Market { asset: usize, side: Side, qty: i64 },
    /// `None` cancels in every asset.
    CancelAll { asset: Option<usize> },
    CancelId(u128),
}

#[derive(Debug, Clone)]
pub struct Fill {
    pub step: u64,
    pub asset: usize,
    pub price_ticks: i64,
    pub lots: i64,
    pub buyer: i64,
    pub seller: i64,
}

#[derive(Debug, Default)]
pub struct StepOut {
    pub rewards: Vec<f64>,
    pub fills: Vec<Fill>,
    /// (agent, order id) of resting orders removed by self-trade prevention.
    pub stp_cancels: Vec<(i64, u128)>,
    pub rejects: Vec<usize>,
    pub done: bool,
}

#[derive(Debug, Clone)]
pub struct OrderInfo {
    pub agent: i64,
    pub asset: usize,
    pub side: Side,
    pub price_ticks: i64,
    pub remaining: i64,
}

struct AssetState {
    cfg: AssetCfg,
    symbol: String,
    book: OrderBook,
    fair_cents: f64,
    last_trade: f64,
    last_mid: f64,
    last_vwap: f64,
    trades: VecDeque<(u64, i64, i64)>,
    house_ids: Vec<Uuid>,
}

struct AgentState {
    cash: i64,
    inv: Vec<i64>,
    open: BTreeSet<u128>,
    prev_equity: f64,
}

pub struct Market {
    pub cfg: Config,
    assets: Vec<AssetState>,
    agents: Vec<AgentState>,
    orders: HashMap<Uuid, OrderInfo>,
    pub t: u64,
    rng: ChaCha8Rng,
    /// Separate stream for action-order shuffling so background randomness does not
    /// depend on how many actions agents submit (keeps counterfactual runs paired).
    shuf_rng: ChaCha8Rng,
    next_id: u128,
    team_of: Vec<i64>,
    // fills / stp events collected during the current step
    fills: Vec<Fill>,
    stp: Vec<(i64, u128)>,
}

fn dec_price(ticks: i64) -> Decimal {
    Decimal::new(ticks, 4)
}
fn dec_qty(lots: i64) -> Decimal {
    Decimal::new(lots, 3)
}

impl Market {
    pub fn new(cfg: Config) -> Result<Self, String> {
        if cfg.assets.is_empty() {
            return Err("need at least one asset".into());
        }
        let team_of = if cfg.team_of.is_empty() {
            (0..cfg.n_agents as i64).collect()
        } else if cfg.team_of.len() == cfg.n_agents {
            cfg.team_of.clone()
        } else {
            return Err("team_of length must equal n_agents".into());
        };
        let mut m = Market {
            cfg,
            assets: vec![],
            agents: vec![],
            orders: HashMap::new(),
            t: 0,
            rng: ChaCha8Rng::seed_from_u64(0),
            shuf_rng: ChaCha8Rng::seed_from_u64(0),
            next_id: 1,
            team_of,
            fills: vec![],
            stp: vec![],
        };
        m.reset(0);
        Ok(m)
    }

    pub fn n_assets(&self) -> usize {
        self.assets.len()
    }
    pub fn n_agents(&self) -> usize {
        self.cfg.n_agents
    }

    pub fn reset(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
        self.shuf_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
        self.t = 0;
        self.next_id = 1;
        self.orders.clear();
        self.fills.clear();
        self.stp.clear();
        self.assets = self
            .cfg
            .assets
            .iter()
            .map(|c| {
                let p_ticks = (c.p0 * 100.0).round() * TICKS_PER_CENT as f64;
                AssetState {
                    cfg: c.clone(),
                    symbol: c.name.clone(),
                    book: OrderBook::new(),
                    fair_cents: c.p0 * 100.0,
                    last_trade: p_ticks,
                    last_mid: p_ticks,
                    last_vwap: p_ticks,
                    trades: VecDeque::new(),
                    house_ids: vec![],
                }
            })
            .collect();
        let na = self.assets.len();
        self.agents = (0..self.cfg.n_agents)
            .map(|a| {
                let inv = (0..na)
                    .map(|k| {
                        let u = self
                            .cfg
                            .init_inventory
                            .get(a)
                            .and_then(|v| v.get(k))
                            .copied()
                            .unwrap_or(0.0);
                        (u * LOTS_PER_UNIT as f64).round() as i64
                    })
                    .collect();
                AgentState {
                    cash: (self.cfg.init_cash * CASH_SCALE).round() as i64,
                    inv,
                    open: BTreeSet::new(),
                    prev_equity: 0.0,
                }
            })
            .collect();
        // Background-only warmup so the book is populated, then restart the clock.
        for k in 0..self.assets.len() {
            self.requote_house(k);
        }
        for _ in 0..WARMUP_STEPS {
            self.background(0);
            self.update_marks();
        }
        self.fills.clear();
        self.stp.clear();
        self.t = 0;
        for a in 0..self.agents.len() {
            self.agents[a].prev_equity = self.equity(a, self.cfg.marking);
        }
    }

    fn fresh_id(&mut self) -> u128 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn owner_key(&self, agent: i64) -> i64 {
        if agent < 0 {
            agent
        } else if self.cfg.team_stp {
            1_000_000 + self.team_of[agent as usize]
        } else {
            1_000 + agent
        }
    }

    // ---- order flow -------------------------------------------------------------

    /// Sends one limit order through the matching core. Returns (order id, resting lots).
    fn submit(
        &mut self,
        agent: i64,
        asset: usize,
        side: Side,
        price_ticks: i64,
        lots: i64,
        ioc: bool,
    ) -> (u128, i64) {
        let id = self.fresh_id();
        let uuid = Uuid::from_u128(id);
        let mut order = Order::new(
            self.owner_key(agent),
            self.assets[asset].symbol.clone(),
            dec_price(price_ticks),
            dec_qty(lots),
            side,
            OrderType::Limit,
        );
        order.id = uuid;
        self.orders.insert(
            uuid,
            OrderInfo { agent, asset, side, price_ticks, remaining: lots },
        );
        let (trades, cancelled) = self.assets[asset].book.process_limit_order(order);
        for tr in trades {
            self.apply_trade(asset, &tr.bid_order_id, &tr.ask_order_id, tr.price_ticks, tr.quantity_lots);
        }
        for c in cancelled {
            if let Some(info) = self.orders.remove(&c.id) {
                if info.agent >= 0 {
                    self.agents[info.agent as usize].open.remove(&c.id.as_u128());
                    self.stp.push((info.agent, c.id.as_u128()));
                }
            }
        }
        let mut resting = 0;
        if let Some(info) = self.orders.get(&uuid) {
            if ioc {
                self.assets[asset].book.cancel_order(uuid);
                self.orders.remove(&uuid);
            } else {
                resting = info.remaining;
                if agent >= 0 {
                    self.agents[agent as usize].open.insert(id);
                }
            }
        }
        (id, resting)
    }

    fn apply_trade(&mut self, asset: usize, bid_id: &Uuid, ask_id: &Uuid, price: i64, lots: i64) {
        let buyer = self.orders.get(bid_id).map(|i| i.agent).unwrap_or(NOISE);
        let seller = self.orders.get(ask_id).map(|i| i.agent).unwrap_or(NOISE);
        for (oid, who) in [(bid_id, buyer), (ask_id, seller)] {
            let done = {
                let info = self.orders.get_mut(oid).expect("trade for unknown order");
                info.remaining -= lots;
                info.remaining <= 0
            };
            if done {
                self.orders.remove(oid);
                if who >= 0 {
                    self.agents[who as usize].open.remove(&oid.as_u128());
                }
            }
        }
        let cost = price * lots;
        if buyer >= 0 {
            let a = &mut self.agents[buyer as usize];
            a.cash -= cost;
            a.inv[asset] += lots;
        }
        if seller >= 0 {
            let a = &mut self.agents[seller as usize];
            a.cash += cost;
            a.inv[asset] -= lots;
        }
        let a = &mut self.assets[asset];
        a.last_trade = price as f64;
        a.trades.push_back((self.t, price, lots));
        self.fills.push(Fill { step: self.t, asset, price_ticks: price, lots, buyer, seller });
    }

    fn cancel_order(&mut self, uuid: Uuid) -> bool {
        let Some(info) = self.orders.get(&uuid).cloned() else { return false };
        self.assets[info.asset].book.cancel_order(uuid);
        self.orders.remove(&uuid);
        if info.agent >= 0 {
            self.agents[info.agent as usize].open.remove(&uuid.as_u128());
        }
        true
    }

    fn open_lots(&self, agent: usize, asset: usize, side: Side) -> i64 {
        self.agents[agent]
            .open
            .iter()
            .filter_map(|id| self.orders.get(&Uuid::from_u128(*id)))
            .filter(|i| i.asset == asset && i.side == side)
            .map(|i| i.remaining)
            .sum()
    }

    fn within_limit(&self, agent: usize, asset: usize, side: Side, lots: i64) -> bool {
        let lim = (self.cfg.pos_limit * LOTS_PER_UNIT as f64) as i64;
        let inv = self.agents[agent].inv[asset];
        match side {
            Side::Bid => inv + self.open_lots(agent, asset, Side::Bid) + lots <= lim,
            Side::Ask => inv - self.open_lots(agent, asset, Side::Ask) - lots >= -lim,
        }
    }

    fn apply_action(&mut self, agent: usize, action: &Action) -> bool {
        match *action {
            Action::Noop => true,
            Action::Limit { asset, side, price_cents, qty } => {
                if asset >= self.assets.len() || qty <= 0 || price_cents <= 0 {
                    return false;
                }
                let lots = qty * LOTS_PER_UNIT;
                if !self.within_limit(agent, asset, side, lots) {
                    return false;
                }
                self.submit(agent as i64, asset, side, price_cents * TICKS_PER_CENT, lots, false);
                true
            }
            Action::Market { asset, side, qty } => {
                if asset >= self.assets.len() || qty <= 0 {
                    return false;
                }
                let lots = qty * LOTS_PER_UNIT;
                if !self.within_limit(agent, asset, side, lots) {
                    return false;
                }
                let px = if side == Side::Bid { i64::MAX / 1_000_000_000 } else { 1 };
                self.submit(agent as i64, asset, side, px, lots, true);
                true
            }
            Action::CancelAll { asset } => {
                let ids: Vec<u128> = self.agents[agent].open.iter().copied().collect();
                for id in ids {
                    let u = Uuid::from_u128(id);
                    let ok = match (self.orders.get(&u), asset) {
                        (Some(i), Some(a)) => i.asset == a,
                        (Some(_), None) => true,
                        _ => false,
                    };
                    if ok {
                        self.cancel_order(u);
                    }
                }
                true
            }
            Action::CancelId(id) => {
                let u = Uuid::from_u128(id);
                match self.orders.get(&u) {
                    Some(i) if i.agent == agent as i64 => self.cancel_order(u),
                    _ => false,
                }
            }
        }
    }

    // ---- background participants -----------------------------------------------

    fn gaussian(&mut self) -> f64 {
        let u1: f64 = self.rng.r#gen::<f64>().max(1e-12);
        let u2: f64 = self.rng.r#gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn poisson(&mut self, lambda: f64) -> usize {
        let l = (-lambda).exp();
        let (mut k, mut p) = (0usize, 1.0f64);
        loop {
            p *= self.rng.r#gen::<f64>();
            if p <= l || k > 50 {
                return k;
            }
            k += 1;
        }
    }

    fn requote_house(&mut self, k: usize) {
        let old = std::mem::take(&mut self.assets[k].house_ids);
        for u in old {
            self.cancel_order(u);
        }
        let c = self.assets[k].cfg.clone();
        let fair = self.assets[k].fair_cents;
        let lots = (c.house_size * LOTS_PER_UNIT as f64).round() as i64;
        let mut ids = vec![];
        for lvl in 0..c.house_levels as i64 {
            let off = c.half_spread_cents + lvl * c.level_gap_cents;
            let bid = ((fair - off as f64).round() as i64).max(1);
            let ask = ((fair + off as f64).round() as i64).max(bid + 1);
            for (side, px) in [(Side::Bid, bid), (Side::Ask, ask)] {
                let (id, resting) = self.submit(HOUSE, k, side, px * TICKS_PER_CENT, lots, false);
                if resting > 0 {
                    ids.push(Uuid::from_u128(id));
                }
            }
        }
        self.assets[k].house_ids = ids;
    }

    fn background(&mut self, _unused: u8) {
        for k in 0..self.assets.len() {
            // noise traders (marketable, immediate-or-cancel)
            let n = self.poisson(self.assets[k].cfg.noise_rate);
            for _ in 0..n {
                let side = if self.rng.gen_bool(0.5) { Side::Bid } else { Side::Ask };
                let lots = self.rng.gen_range(1..=self.assets[k].cfg.noise_size.max(1)) * LOTS_PER_UNIT;
                let px = if side == Side::Bid { i64::MAX / 1_000_000_000 } else { 1 };
                self.submit(NOISE, k, side, px, lots, true);
            }
            // fair value moves, house requotes
            let shock = self.gaussian() * self.assets[k].cfg.vol_cents;
            let a = &mut self.assets[k];
            a.fair_cents = (a.fair_cents + shock).max(100.0);
            self.requote_house(k);
        }
    }

    // ---- marks, equity ----------------------------------------------------------

    fn update_marks(&mut self) {
        let t = self.t;
        let w = self.cfg.vwap_window;
        for a in &mut self.assets {
            if let (Some(b), Some(s)) = (a.book.best_bid_price(), a.book.best_ask_price()) {
                use rust_decimal::prelude::ToPrimitive;
                let (b, s) = (b.to_f64().unwrap_or(0.0), s.to_f64().unwrap_or(0.0));
                a.last_mid = (b + s) / 2.0 * 10_000.0;
            }
            while a.trades.front().is_some_and(|(st, _, _)| st + w <= t) {
                a.trades.pop_front();
            }
            let (mut pv, mut v) = (0f64, 0f64);
            for (_, p, q) in &a.trades {
                pv += *p as f64 * *q as f64;
                v += *q as f64;
            }
            if v > 0.0 {
                a.last_vwap = pv / v;
            }
        }
    }

    /// Mark price in core ticks (f64) for `asset` under `rule`.
    pub fn mark(&self, asset: usize, rule: MarkRule) -> f64 {
        let a = &self.assets[asset];
        match rule {
            MarkRule::Last => a.last_trade,
            MarkRule::Mid => a.last_mid,
            MarkRule::Vwap => a.last_vwap,
            MarkRule::Fair => a.fair_cents * TICKS_PER_CENT as f64,
        }
    }

    /// Marked equity in dollars.
    pub fn equity(&self, agent: usize, rule: MarkRule) -> f64 {
        let a = &self.agents[agent];
        let mut e = a.cash as f64 / CASH_SCALE;
        for k in 0..self.assets.len() {
            e += a.inv[k] as f64 / LOTS_PER_UNIT as f64 * self.mark(k, rule) / 10_000.0;
        }
        e
    }

    // ---- step -------------------------------------------------------------------

    pub fn step(&mut self, actions: &[(usize, Action)]) -> StepOut {
        self.fills.clear();
        self.stp.clear();
        // Agents act in a seeded random order; one agent's actions stay in submitted order
        // (so e.g. cancel-then-requote is atomic from that agent's point of view).
        let mut agents_order: Vec<usize> = vec![];
        for (a, _) in actions {
            if !agents_order.contains(a) {
                agents_order.push(*a);
            }
        }
        if self.cfg.shuffle_actions {
            for i in (1..agents_order.len()).rev() {
                let j = self.shuf_rng.gen_range(0..=i);
                agents_order.swap(i, j);
            }
        }
        let mut rejects = vec![];
        for ag in agents_order {
            for (agent, act) in actions.iter().filter(|(a, _)| *a == ag) {
                if *agent >= self.agents.len() || !self.apply_action(*agent, act) {
                    rejects.push(*agent);
                }
            }
        }
        self.background(0);
        self.t += 1;
        self.update_marks();

        let mut rewards = Vec::with_capacity(self.agents.len());
        for a in 0..self.agents.len() {
            let eq = self.equity(a, self.cfg.marking);
            let mut r = eq - self.agents[a].prev_equity;
            if self.cfg.inv_penalty > 0.0 {
                let mut gross = 0.0;
                for k in 0..self.assets.len() {
                    gross += (self.agents[a].inv[k] as f64 / LOTS_PER_UNIT as f64).abs()
                        * self.mark(k, self.cfg.marking)
                        / 10_000.0;
                }
                r -= self.cfg.inv_penalty * gross;
            }
            self.agents[a].prev_equity = eq;
            rewards.push(r * self.cfg.reward_scale);
        }
        StepOut {
            rewards,
            fills: std::mem::take(&mut self.fills),
            stp_cancels: std::mem::take(&mut self.stp),
            rejects,
            done: self.t >= self.cfg.episode_len,
        }
    }

    // ---- views ------------------------------------------------------------------

    /// L2 levels as (price_cents, units), best first.
    pub fn levels(&self, asset: usize, side: Side, depth: usize) -> Vec<(f64, f64)> {
        self.assets[asset]
            .book
            .aggregated_levels(side)
            .into_iter()
            .take(depth)
            .map(|(p, q)| (p as f64 / TICKS_PER_CENT as f64, q as f64 / LOTS_PER_UNIT as f64))
            .collect()
    }

    pub fn account(&self, agent: usize) -> (f64, Vec<f64>) {
        let a = &self.agents[agent];
        (
            a.cash as f64 / CASH_SCALE,
            a.inv.iter().map(|l| *l as f64 / LOTS_PER_UNIT as f64).collect(),
        )
    }

    /// (order id, asset, side, price_cents, remaining units)
    pub fn open_orders(&self, agent: usize) -> Vec<(u128, usize, Side, f64, f64)> {
        self.agents[agent]
            .open
            .iter()
            .filter_map(|id| {
                self.orders.get(&Uuid::from_u128(*id)).map(|i| {
                    (
                        *id,
                        i.asset,
                        i.side,
                        i.price_ticks as f64 / TICKS_PER_CENT as f64,
                        i.remaining as f64 / LOTS_PER_UNIT as f64,
                    )
                })
            })
            .collect()
    }

    pub fn fair_cents(&self, asset: usize) -> f64 {
        self.assets[asset].fair_cents
    }

    /// Reference price in cents used to express relative price offsets: last valid mid.
    pub fn ref_cents(&self, asset: usize) -> i64 {
        (self.assets[asset].last_mid / TICKS_PER_CENT as f64).round() as i64
    }

    pub fn obs_dim(&self) -> usize {
        self.assets.len() * (4 * self.cfg.obs_depth + 5) + 3
    }

    /// Flat float observation for one agent (own inventory/orders only).
    pub fn obs(&self, agent: usize) -> Vec<f32> {
        let d = self.cfg.obs_depth;
        let mut v = Vec::with_capacity(self.obs_dim());
        for k in 0..self.assets.len() {
            let r = self.assets[k].last_mid / TICKS_PER_CENT as f64;
            let p0c = self.assets[k].cfg.p0 * 100.0;
            for side in [Side::Bid, Side::Ask] {
                let lv = self.levels(k, side, d);
                for i in 0..d {
                    match lv.get(i) {
                        Some((p, q)) => {
                            v.push(((p - r) / 20.0) as f32);
                            v.push((q / 10.0) as f32);
                        }
                        None => {
                            v.push(0.0);
                            v.push(0.0);
                        }
                    }
                }
            }
            let lt = self.assets[k].last_trade / TICKS_PER_CENT as f64;
            v.push(((lt - r) / 20.0) as f32);
            v.push(((r / p0c - 1.0) * 10.0) as f32);
            v.push((self.agents[agent].inv[k] as f64 / LOTS_PER_UNIT as f64 / 50.0) as f32);
            v.push((self.open_lots(agent, k, Side::Bid) as f64 / LOTS_PER_UNIT as f64 / 10.0) as f32);
            v.push((self.open_lots(agent, k, Side::Ask) as f64 / LOTS_PER_UNIT as f64 / 10.0) as f32);
        }
        let eq = self.equity(agent, self.cfg.marking);
        v.push(((self.agents[agent].cash as f64 / CASH_SCALE) / self.cfg.init_cash - 1.0) as f32);
        v.push(((eq / self.cfg.init_cash - 1.0) * 10.0) as f32);
        v.push((self.t as f64 / self.cfg.episode_len as f64) as f32);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg2() -> Config {
        Config { n_agents: 2, team_of: vec![0, 0], ..Default::default() }
    }

    #[test]
    fn deterministic_given_seed() {
        let run = |seed| {
            let mut m = Market::new(cfg2()).unwrap();
            m.reset(seed);
            let mut trace = vec![];
            for i in 0..50 {
                let r = ref_c(&m);
                let side = if i % 2 == 0 { Side::Bid } else { Side::Ask };
                let out = m.step(&[
                    (0, Action::Limit { asset: 0, side, price_cents: r, qty: 2 }),
                    (1, Action::Market { asset: 0, side: Side::Bid, qty: 1 }),
                ]);
                trace.push((out.rewards.clone(), out.fills.len(), m.fair_cents(0).to_bits()));
            }
            trace
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    fn ref_c(m: &Market) -> i64 {
        m.ref_cents(0)
    }

    #[test]
    fn position_limits_hold_under_aggressive_quoting() {
        let mut m = Market::new(cfg2()).unwrap();
        m.reset(1);
        for _ in 0..100 {
            let r = ref_c(&m);
            m.step(&[
                (0, Action::Limit { asset: 0, side: Side::Bid, price_cents: r + 2, qty: 3 }),
                (1, Action::Limit { asset: 0, side: Side::Ask, price_cents: r - 2, qty: 3 }),
            ]);
        }
        let (_c0, i0) = m.account(0);
        let (_c1, i1) = m.account(1);
        assert!(i0[0].abs() <= m.cfg.pos_limit && i1[0].abs() <= m.cfg.pos_limit);
    }

    #[test]
    fn stp_per_bot_allows_teammate_wash_but_team_stp_blocks() {
        for team_stp in [false, true] {
            let mut c = Config { team_stp, ..cfg2() };
            c.assets[0].house_levels = 0; // empty book: only the teammates' orders exist
            let mut m = Market::new(c).unwrap();
            m.reset(3);
            // Teammate 0 rests a far-away ask that nothing else touches, teammate 1 lifts it.
            let px = ref_c(&m) + 500;
            m.step(&[(0, Action::Limit { asset: 0, side: Side::Ask, price_cents: px, qty: 2 })]);
            let out = m.step(&[(1, Action::Limit { asset: 0, side: Side::Bid, price_cents: px, qty: 2 })]);
            let wash = out.fills.iter().any(|f| f.buyer == 1 && f.seller == 0);
            assert_eq!(wash, !team_stp, "team_stp={team_stp}");
            if team_stp {
                assert!(out.stp_cancels.iter().any(|(a, _)| *a == 0));
            }
        }
    }

    #[test]
    fn rejects_over_position_limit() {
        let mut m = Market::new(Config { pos_limit: 5.0, ..cfg2() }).unwrap();
        m.reset(0);
        let out = m.step(&[(0, Action::Market { asset: 0, side: Side::Bid, qty: 6 })]);
        assert_eq!(out.rejects, vec![0]);
    }
}
