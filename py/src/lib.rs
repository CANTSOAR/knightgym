pub mod sim;

use knightgym_core::models::Side;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sim::{Action, Config, MarkRule, Market};

fn side_of(s: u8) -> Side {
    if s == 0 { Side::Bid } else { Side::Ask }
}

/// Thin binding over `sim::Market`.
#[pyclass(name = "Market", unsendable)]
struct PyMarket {
    inner: Market,
}

#[pymethods]
impl PyMarket {
    /// `config_json`: JSON for `sim::Config` (see Python `knightgym.config`).
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let cfg: Config = serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: Market::new(cfg).map_err(PyValueError::new_err)? })
    }

    fn reset(&mut self, seed: u64) {
        self.inner.reset(seed);
    }

    /// actions: list of (agent, kind, asset, side, price_cents, qty)
    /// kind: 0 noop, 1 limit, 2 market, 3 cancel_all (asset=-1 for every asset), 4 cancel_id (id in `qty`).
    /// side: 0 bid, 1 ask. Returns a dict with rewards, fills, stp_cancels, rejects, done.
    fn step<'py>(&mut self, py: Python<'py>, actions: Vec<(usize, u8, i64, u8, i64, i64)>) -> PyResult<Bound<'py, PyDict>> {
        let mut acts = Vec::with_capacity(actions.len());
        for (agent, kind, asset, side, price, qty) in actions {
            let a = match kind {
                0 => Action::Noop,
                1 => Action::Limit { asset: asset as usize, side: side_of(side), price_cents: price, qty },
                2 => Action::Market { asset: asset as usize, side: side_of(side), qty },
                3 => Action::CancelAll { asset: if asset < 0 { None } else { Some(asset as usize) } },
                4 => Action::CancelId(qty as u128),
                k => return Err(PyValueError::new_err(format!("bad action kind {k}"))),
            };
            acts.push((agent, a));
        }
        let out = self.inner.step(&acts);
        let d = PyDict::new(py);
        d.set_item("rewards", out.rewards)?;
        let fills: Vec<(u64, usize, f64, f64, i64, i64)> = out
            .fills
            .iter()
            .map(|f| {
                (
                    f.step,
                    f.asset,
                    f.price_ticks as f64 / 10_000.0,
                    f.lots as f64 / 1000.0,
                    f.buyer,
                    f.seller,
                )
            })
            .collect();
        d.set_item("fills", fills)?;
        d.set_item("stp_cancels", out.stp_cancels.iter().map(|(a, id)| (*a, *id as u64)).collect::<Vec<_>>())?;
        d.set_item("rejects", out.rejects)?;
        d.set_item("done", out.done)?;
        Ok(d)
    }

    fn obs<'py>(&self, py: Python<'py>, agent: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if agent >= self.inner.n_agents() {
            return Err(PyValueError::new_err("bad agent"));
        }
        Ok(self.inner.obs(agent).into_pyarray(py))
    }

    fn obs_dim(&self) -> usize {
        self.inner.obs_dim()
    }

    /// (bids, asks), each a list of (price_cents, units), best first.
    fn book(&self, asset: usize, depth: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        (self.inner.levels(asset, Side::Bid, depth), self.inner.levels(asset, Side::Ask, depth))
    }

    /// (cash dollars, [inventory units per asset])
    fn account(&self, agent: usize) -> (f64, Vec<f64>) {
        self.inner.account(agent)
    }

    fn open_orders(&self, agent: usize) -> Vec<(u64, usize, u8, f64, f64)> {
        self.inner
            .open_orders(agent)
            .into_iter()
            .map(|(id, a, s, p, q)| (id as u64, a, if s == Side::Bid { 0 } else { 1 }, p, q))
            .collect()
    }

    /// Mark price in dollars per asset under "last" | "mid" | "vwap".
    fn marks(&self, rule: &str) -> PyResult<Vec<f64>> {
        let r = MarkRule::parse(rule).ok_or_else(|| PyValueError::new_err("rule must be last|mid|vwap"))?;
        Ok((0..self.inner.n_assets()).map(|k| self.inner.mark(k, r) / 10_000.0).collect())
    }

    fn equity(&self, agent: usize, rule: &str) -> PyResult<f64> {
        let r = MarkRule::parse(rule).ok_or_else(|| PyValueError::new_err("rule must be last|mid|vwap"))?;
        Ok(self.inner.equity(agent, r))
    }

    fn ref_cents(&self, asset: usize) -> i64 {
        self.inner.ref_cents(asset)
    }

    fn fair_cents(&self, asset: usize) -> f64 {
        self.inner.fair_cents(asset)
    }

    #[getter]
    fn t(&self) -> u64 {
        self.inner.t
    }

    #[getter]
    fn n_agents(&self) -> usize {
        self.inner.n_agents()
    }

    #[getter]
    fn n_assets(&self) -> usize {
        self.inner.n_assets()
    }
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMarket>()?;
    Ok(())
}
