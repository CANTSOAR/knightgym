use crate::models::{BotId, Order, PriceTicks, Side, Trade, price_ticks_to_decimal};
use rust_decimal::Decimal;
use serde::ser::{SerializeMap, SerializeStruct};
use serde::{Serialize, Serializer};
use slab::Slab;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Debug, Clone)]
struct OrderNode {
    order: Order,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug, Clone, Default)]
struct PriceLevel {
    head: Option<usize>,
    tail: Option<usize>,
    len: usize,
}

#[derive(Debug, Clone, Copy)]
struct ActiveOrderRef {
    side: Side,
    price_ticks: PriceTicks,
    slot: usize,
}

#[derive(Debug, Clone)]
pub struct OrderBook {
    bids: BTreeMap<PriceTicks, PriceLevel>,
    asks: BTreeMap<PriceTicks, PriceLevel>,
    arena: Slab<OrderNode>,
    active_orders: HashMap<Uuid, ActiveOrderRef>,
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            arena: Slab::new(),
            active_orders: HashMap::new(),
        }
    }

    pub fn process_limit_order(&mut self, mut order: Order) -> (Vec<Trade>, Vec<Order>) {
        let mut trades = Vec::new();
        let mut cancelled_orders = Vec::new();

        match order.side {
            Side::Bid => self.match_bid(&mut order, &mut trades, &mut cancelled_orders),
            Side::Ask => self.match_ask(&mut order, &mut trades, &mut cancelled_orders),
        }

        if order.quantity_lots > 0 {
            self.add_order_to_book(order);
        }

        (trades, cancelled_orders)
    }

    pub fn add_order_to_book(&mut self, mut order: Order) {
        if order.price_ticks == 0 {
            order.price_ticks =
                crate::models::decimal_to_price_ticks(order.price).unwrap_or_default();
        }
        if order.quantity_lots == 0 && order.quantity > Decimal::ZERO {
            order.quantity_lots =
                crate::models::decimal_to_quantity_lots(order.quantity).unwrap_or_default();
        }

        let price_ticks = order.price_ticks;
        let side = order.side;
        let order_id = order.id;
        let slot = self.arena.insert(OrderNode {
            order,
            prev: None,
            next: None,
        });

        self.append_to_level(side, price_ticks, slot);
        self.active_orders.insert(
            order_id,
            ActiveOrderRef {
                side,
                price_ticks,
                slot,
            },
        );
    }

    pub fn cancel_order(&mut self, order_id: Uuid) -> Option<Order> {
        let active = self.active_orders.remove(&order_id)?;
        self.unlink_slot(active.side, active.price_ticks, active.slot);
        self.arena.try_remove(active.slot).map(|node| node.order)
    }

    pub fn cancel_all_for_owner(&mut self, owner_id: BotId) -> Vec<Order> {
        let order_ids: Vec<Uuid> = self
            .active_orders
            .iter()
            .filter_map(|(order_id, active)| {
                self.arena
                    .get(active.slot)
                    .filter(|node| node.order.owner_id == owner_id)
                    .map(|_| *order_id)
            })
            .collect();

        order_ids
            .into_iter()
            .filter_map(|order_id| self.cancel_order(order_id))
            .collect()
    }

    pub fn cancel_all_for_owners(&mut self, owner_ids: &HashSet<BotId>) -> Vec<Order> {
        let order_ids: Vec<Uuid> = self
            .active_orders
            .iter()
            .filter_map(|(order_id, active)| {
                self.arena
                    .get(active.slot)
                    .filter(|node| owner_ids.contains(&node.order.owner_id))
                    .map(|_| *order_id)
            })
            .collect();

        order_ids
            .into_iter()
            .filter_map(|order_id| self.cancel_order(order_id))
            .collect()
    }

    pub fn cancel_crossing_orders_for_owner(
        &mut self,
        incoming_side: Side,
        owner_id: BotId,
        limit_price_ticks: PriceTicks,
    ) -> Vec<Order> {
        let order_ids: Vec<Uuid> = self
            .active_orders
            .iter()
            .filter_map(|(order_id, active)| {
                let is_crossing = match incoming_side {
                    Side::Bid => active.side == Side::Ask && active.price_ticks <= limit_price_ticks,
                    Side::Ask => active.side == Side::Bid && active.price_ticks >= limit_price_ticks,
                };

                if !is_crossing {
                    return None;
                }

                self.arena
                    .get(active.slot)
                    .filter(|node| node.order.owner_id == owner_id)
                    .map(|_| *order_id)
            })
            .collect();

        order_ids
            .into_iter()
            .filter_map(|order_id| self.cancel_order(order_id))
            .collect()
    }

    pub fn clear_house_orders(&mut self, house_id: BotId) {
        let _ = self.cancel_all_for_owner(house_id);
    }

    pub fn clear(&mut self) -> usize {
        let count = self.active_orders.len();
        self.bids.clear();
        self.asks.clear();
        self.arena.clear();
        self.active_orders.clear();
        count
    }

    pub fn into_active_orders(mut self) -> Vec<Order> {
        let mut orders = Vec::with_capacity(self.active_orders.len());
        for (_, active) in self.active_orders.drain() {
            if let Some(node) = self.arena.try_remove(active.slot) {
                orders.push(node.order);
            }
        }
        orders
    }

    pub fn active_order_count(&self) -> usize {
        self.active_orders.len()
    }

    pub fn level_count(&self, side: Side) -> usize {
        self.side_levels(side).len()
    }

    pub fn total_quantity(&self, side: Side) -> Decimal {
        self.active_orders_by_side(side)
            .into_iter()
            .fold(Decimal::ZERO, |acc, order| acc + order.quantity)
    }

    pub fn best_bid_price(&self) -> Option<Decimal> {
        self.best_price_ticks(Side::Bid).map(price_ticks_to_decimal)
    }

    pub fn best_ask_price(&self) -> Option<Decimal> {
        self.best_price_ticks(Side::Ask).map(price_ticks_to_decimal)
    }

    pub fn active_orders_by_side(&self, side: Side) -> Vec<Order> {
        let mut out = Vec::new();
        match side {
            Side::Bid => {
                for (price_ticks, level) in self.bids.iter().rev() {
                    self.collect_level_orders(*price_ticks, side, level, &mut out);
                }
            }
            Side::Ask => {
                for (price_ticks, level) in &self.asks {
                    self.collect_level_orders(*price_ticks, side, level, &mut out);
                }
            }
        }
        out
    }

    pub fn active_orders(&self) -> Vec<Order> {
        let mut out = self.active_orders_by_side(Side::Bid);
        out.extend(self.active_orders_by_side(Side::Ask));
        out
    }

    pub fn snapshot_levels(&self, side: Side) -> Vec<(Decimal, Vec<Order>)> {
        let mut levels = Vec::new();
        match side {
            Side::Bid => {
                for (price_ticks, level) in self.bids.iter().rev() {
                    let mut orders = Vec::with_capacity(level.len);
                    self.collect_level_orders(*price_ticks, side, level, &mut orders);
                    if !orders.is_empty() {
                        levels.push((price_ticks_to_decimal(*price_ticks), orders));
                    }
                }
            }
            Side::Ask => {
                for (price_ticks, level) in &self.asks {
                    let mut orders = Vec::with_capacity(level.len);
                    self.collect_level_orders(*price_ticks, side, level, &mut orders);
                    if !orders.is_empty() {
                        levels.push((price_ticks_to_decimal(*price_ticks), orders));
                    }
                }
            }
        }
        levels
    }

    pub fn aggregated_levels(&self, side: Side) -> Vec<(PriceTicks, i64)> {
        let mut levels = Vec::new();
        match side {
            Side::Bid => {
                for (price_ticks, level) in self.bids.iter().rev() {
                    let quantity_lots = self.level_quantity_lots(side, *price_ticks, level);
                    if quantity_lots > 0 {
                        levels.push((*price_ticks, quantity_lots));
                    }
                }
            }
            Side::Ask => {
                for (price_ticks, level) in &self.asks {
                    let quantity_lots = self.level_quantity_lots(side, *price_ticks, level);
                    if quantity_lots > 0 {
                        levels.push((*price_ticks, quantity_lots));
                    }
                }
            }
        }
        levels
    }

    pub fn aggregated_quantity_at(&self, side: Side, price_ticks: PriceTicks) -> i64 {
        self.side_levels(side)
            .get(&price_ticks)
            .map(|level| self.level_quantity_lots(side, price_ticks, level))
            .unwrap_or_default()
    }

    pub fn contains_order(&self, order_id: Uuid) -> bool {
        self.active_orders.contains_key(&order_id)
    }

    #[cfg(test)]
    fn arena_len(&self) -> usize {
        self.arena.len()
    }

    #[cfg(test)]
    fn level_len_at(&self, side: Side, price_ticks: PriceTicks) -> usize {
        self.side_levels(side)
            .get(&price_ticks)
            .map(|level| level.len)
            .unwrap_or(0)
    }

    fn match_bid(&mut self, bid: &mut Order, trades: &mut Vec<Trade>, cancelled: &mut Vec<Order>) {
        while bid.quantity_lots > 0 {
            let best_ask_price = match self.best_price_ticks(Side::Ask) {
                Some(price) => price,
                None => break,
            };

            if best_ask_price > bid.price_ticks {
                break;
            }

            let maker_owner = match self.peek_best_owner(Side::Ask, best_ask_price) {
                Some(owner_id) => owner_id,
                None => continue,
            };

            if maker_owner == bid.owner_id {
                if let Some(order) = self.cancel_front_order(Side::Ask, best_ask_price) {
                    cancelled.push(order);
                }
                continue;
            }

            self.execute_match(best_ask_price, bid, trades, Side::Ask);
        }
    }

    fn match_ask(&mut self, ask: &mut Order, trades: &mut Vec<Trade>, cancelled: &mut Vec<Order>) {
        while ask.quantity_lots > 0 {
            let best_bid_price = match self.best_price_ticks(Side::Bid) {
                Some(price) => price,
                None => break,
            };

            if best_bid_price < ask.price_ticks {
                break;
            }

            let maker_owner = match self.peek_best_owner(Side::Bid, best_bid_price) {
                Some(owner_id) => owner_id,
                None => continue,
            };

            if maker_owner == ask.owner_id {
                if let Some(order) = self.cancel_front_order(Side::Bid, best_bid_price) {
                    cancelled.push(order);
                }
                continue;
            }

            self.execute_match(best_bid_price, ask, trades, Side::Bid);
        }
    }

    fn execute_match(
        &mut self,
        price_ticks: PriceTicks,
        incoming_order: &mut Order,
        trades: &mut Vec<Trade>,
        target_side: Side,
    ) {
        let Some((_slot, mut book_order)) = self.pop_front_active_order(target_side, price_ticks)
        else {
            return;
        };

        let trade_qty_lots = incoming_order.quantity_lots.min(book_order.quantity_lots);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        trades.push(Trade::from_fixed(
            if target_side == Side::Bid {
                book_order.id
            } else {
                incoming_order.id
            },
            if target_side == Side::Ask {
                book_order.id
            } else {
                incoming_order.id
            },
            if target_side == Side::Bid {
                book_order.owner_id
            } else {
                incoming_order.owner_id
            },
            if target_side == Side::Ask {
                book_order.owner_id
            } else {
                incoming_order.owner_id
            },
            incoming_order.symbol.clone(),
            price_ticks,
            trade_qty_lots,
            now,
            0,
        ));

        incoming_order.set_quantity_lots(incoming_order.quantity_lots - trade_qty_lots);
        book_order.set_quantity_lots(book_order.quantity_lots - trade_qty_lots);

        if book_order.quantity_lots > 0 {
            let side = book_order.side;
            let order_id = book_order.id;
            let reinserted_slot = self.arena.insert(OrderNode {
                order: book_order,
                prev: None,
                next: None,
            });
            self.prepend_to_level(side, price_ticks, reinserted_slot);
            self.active_orders.insert(
                order_id,
                ActiveOrderRef {
                    side,
                    price_ticks,
                    slot: reinserted_slot,
                },
            );
        }
    }

    fn cancel_front_order(&mut self, side: Side, price_ticks: PriceTicks) -> Option<Order> {
        let (slot, order) = self.pop_front_active_order(side, price_ticks)?;
        let _ = self.arena.try_remove(slot);
        Some(order)
    }

    fn pop_front_active_order(
        &mut self,
        side: Side,
        price_ticks: PriceTicks,
    ) -> Option<(usize, Order)> {
        let slot = self.pop_front_from_level(side, price_ticks)?;
        let node = self.arena.try_remove(slot)?;
        self.active_orders.remove(&node.order.id);
        Some((slot, node.order))
    }

    fn peek_best_owner(&self, side: Side, price_ticks: PriceTicks) -> Option<BotId> {
        let level = self.side_levels(side).get(&price_ticks)?;
        let slot = level.head?;
        let node = self.arena.get(slot)?;
        self.active_orders
            .get(&node.order.id)
            .filter(|active| {
                active.side == side && active.price_ticks == price_ticks && active.slot == slot
            })
            .map(|_| node.order.owner_id)
    }

    fn active_order_clone(&self, slot: usize) -> Option<Order> {
        let node = self.arena.get(slot)?;
        self.active_orders
            .get(&node.order.id)
            .filter(|active| active.side == node.order.side && active.slot == slot)
            .map(|_| node.order.clone())
    }

    fn slot_is_active(&self, slot: usize) -> bool {
        self.active_order_clone(slot).is_some()
    }

    fn best_price_ticks(&self, side: Side) -> Option<PriceTicks> {
        match side {
            Side::Bid => self.bids.iter().rev().next().map(|(price, _)| *price),
            Side::Ask => self.asks.iter().next().map(|(price, _)| *price),
        }
    }

    fn collect_level_orders(
        &self,
        _price_ticks: PriceTicks,
        _side: Side,
        level: &PriceLevel,
        out: &mut Vec<Order>,
    ) {
        let mut cursor = level.head;
        while let Some(slot) = cursor {
            let Some(node) = self.arena.get(slot) else {
                break;
            };
            if self.slot_is_active(slot) {
                out.push(node.order.clone());
            }
            cursor = node.next;
        }
    }

    fn level_quantity_lots(&self, _side: Side, _price_ticks: PriceTicks, level: &PriceLevel) -> i64 {
        let mut cursor = level.head;
        let mut sum = 0;
        while let Some(slot) = cursor {
            let Some(node) = self.arena.get(slot) else {
                break;
            };
            if self.slot_is_active(slot) {
                sum += node.order.quantity_lots;
            }
            cursor = node.next;
        }
        sum
    }

    fn append_to_level(&mut self, side: Side, price_ticks: PriceTicks, slot: usize) {
        let previous_tail = self
            .side_levels(side)
            .get(&price_ticks)
            .and_then(|level| level.tail);

        {
            let node = &mut self.arena[slot];
            node.prev = previous_tail;
            node.next = None;
        }

        if let Some(prev_slot) = previous_tail {
            self.arena[prev_slot].next = Some(slot);
        }

        let levels = self.side_levels_mut(side);
        let level = levels.entry(price_ticks).or_default();
        if level.head.is_none() {
            level.head = Some(slot);
        }
        level.tail = Some(slot);
        level.len += 1;
    }

    fn prepend_to_level(&mut self, side: Side, price_ticks: PriceTicks, slot: usize) {
        let previous_head = self
            .side_levels(side)
            .get(&price_ticks)
            .and_then(|level| level.head);

        {
            let node = &mut self.arena[slot];
            node.prev = None;
            node.next = previous_head;
        }

        if let Some(next_slot) = previous_head {
            self.arena[next_slot].prev = Some(slot);
        }

        let levels = self.side_levels_mut(side);
        let level = levels.entry(price_ticks).or_default();
        if level.tail.is_none() {
            level.tail = Some(slot);
        }
        level.head = Some(slot);
        level.len += 1;
    }

    fn pop_front_from_level(&mut self, side: Side, price_ticks: PriceTicks) -> Option<usize> {
        let head_slot = self.side_levels(side).get(&price_ticks)?.head?;
        self.unlink_slot(side, price_ticks, head_slot);
        Some(head_slot)
    }

    fn unlink_slot(&mut self, side: Side, price_ticks: PriceTicks, slot: usize) {
        let (prev, next) = match self.arena.get(slot) {
            Some(node) => (node.prev, node.next),
            None => return,
        };

        if let Some(prev_slot) = prev {
            if let Some(prev_node) = self.arena.get_mut(prev_slot) {
                prev_node.next = next;
            }
        }
        if let Some(next_slot) = next {
            if let Some(next_node) = self.arena.get_mut(next_slot) {
                next_node.prev = prev;
            }
        }

        let remove_level = {
            let levels = self.side_levels_mut(side);
            let Some(level) = levels.get_mut(&price_ticks) else {
                return;
            };
            if level.head == Some(slot) {
                level.head = next;
            }
            if level.tail == Some(slot) {
                level.tail = prev;
            }
            level.len = level.len.saturating_sub(1);
            level.len == 0
        };
        if remove_level {
            self.side_levels_mut(side).remove(&price_ticks);
        }

        if let Some(node) = self.arena.get_mut(slot) {
            node.prev = None;
            node.next = None;
        }
    }

    fn side_levels(&self, side: Side) -> &BTreeMap<PriceTicks, PriceLevel> {
        match side {
            Side::Bid => &self.bids,
            Side::Ask => &self.asks,
        }
    }

    fn side_levels_mut(&mut self, side: Side) -> &mut BTreeMap<PriceTicks, PriceLevel> {
        match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        }
    }
}

impl Serialize for OrderBook {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("OrderBook", 2)?;
        state.serialize_field("bids", &SideSnapshot(self.snapshot_levels(Side::Bid)))?;
        state.serialize_field("asks", &SideSnapshot(self.snapshot_levels(Side::Ask)))?;
        state.end()
    }
}

struct SideSnapshot(Vec<(Decimal, Vec<Order>)>);

impl Serialize for SideSnapshot {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (price, orders) in &self.0 {
            map.serialize_entry(&price.to_string(), orders)?;
        }
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::OrderType;
    use rust_decimal_macros::dec;

    fn limit_order(owner_id: BotId, side: Side, price: Decimal, quantity: Decimal) -> Order {
        let mut order = Order::new(
            owner_id,
            "ABC".to_string(),
            price,
            quantity,
            side,
            OrderType::Limit,
        );
        order.refresh_fixed_fields().unwrap();
        order
    }

    #[test]
    fn partial_fills_preserve_fixed_point_quantities() {
        let mut book = OrderBook::new();
        book.add_order_to_book(limit_order(11, Side::Ask, dec!(10.25), dec!(5.000)));

        let incoming = limit_order(12, Side::Bid, dec!(10.25), dec!(1.250));
        let (trades, _) = book.process_limit_order(incoming);

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity_lots, 1_250);
        assert_eq!(trades[0].quantity.to_string(), "1.250");
        let remaining = book.active_orders_by_side(Side::Ask)[0].quantity_lots;
        assert_eq!(remaining, 3_750);
    }

    #[test]
    fn cancel_order_reclaims_slot_immediately() {
        let mut book = OrderBook::new();
        let price_ticks = crate::models::decimal_to_price_ticks(dec!(10.25)).unwrap();

        let order_a = limit_order(21, Side::Bid, dec!(10.25), dec!(1.000));
        let order_b = limit_order(22, Side::Bid, dec!(10.25), dec!(1.000));
        let order_c = limit_order(23, Side::Bid, dec!(10.25), dec!(1.000));
        let cancel_id = order_b.id;

        book.add_order_to_book(order_a);
        book.add_order_to_book(order_b);
        book.add_order_to_book(order_c);

        assert_eq!(book.arena_len(), 3);
        assert_eq!(book.level_len_at(Side::Bid, price_ticks), 3);

        let cancelled = book.cancel_order(cancel_id).unwrap();

        assert_eq!(cancelled.id, cancel_id);
        assert_eq!(book.arena_len(), 2);
        assert_eq!(book.level_len_at(Side::Bid, price_ticks), 2);
        let remaining_ids: Vec<Uuid> = book
            .active_orders_by_side(Side::Bid)
            .into_iter()
            .map(|order| order.id)
            .collect();
        assert_eq!(remaining_ids.len(), 2);
        assert!(!remaining_ids.contains(&cancel_id));
    }

    #[test]
    fn cancel_order_preserves_fifo_at_level() {
        let mut book = OrderBook::new();
        let first = limit_order(31, Side::Ask, dec!(10.25), dec!(1.000));
        let middle = limit_order(32, Side::Ask, dec!(10.25), dec!(1.000));
        let last = limit_order(33, Side::Ask, dec!(10.25), dec!(1.000));
        let middle_id = middle.id;
        let first_id = first.id;
        let last_id = last.id;

        book.add_order_to_book(first);
        book.add_order_to_book(middle);
        book.add_order_to_book(last);
        let _ = book.cancel_order(middle_id);

        let incoming = limit_order(99, Side::Bid, dec!(10.25), dec!(1.000));
        let (trades, _) = book.process_limit_order(incoming);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].ask_order_id, first_id);

        let incoming = limit_order(100, Side::Bid, dec!(10.25), dec!(1.000));
        let (trades, _) = book.process_limit_order(incoming);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].ask_order_id, last_id);
    }

    #[test]
    fn cancel_all_reclaims_all_slots_immediately() {
        let mut book = OrderBook::new();
        book.add_order_to_book(limit_order(41, Side::Bid, dec!(10.25), dec!(1.000)));
        book.add_order_to_book(limit_order(41, Side::Bid, dec!(10.26), dec!(1.000)));
        book.add_order_to_book(limit_order(41, Side::Ask, dec!(10.27), dec!(1.000)));

        let cancelled = book.cancel_all_for_owner(41);
        assert_eq!(cancelled.len(), 3);
        assert_eq!(book.active_order_count(), 0);
        assert_eq!(book.arena_len(), 0);
        assert_eq!(book.level_count(Side::Bid), 0);
        assert_eq!(book.level_count(Side::Ask), 0);
    }

    #[test]
    fn empty_level_removed_when_last_order_cancelled() {
        let mut book = OrderBook::new();
        let order = limit_order(51, Side::Ask, dec!(10.25), dec!(1.000));
        let price_ticks = order.price_ticks;
        let order_id = order.id;
        book.add_order_to_book(order);

        let _ = book.cancel_order(order_id);

        assert_eq!(book.level_len_at(Side::Ask, price_ticks), 0);
        assert_eq!(book.level_count(Side::Ask), 0);
        assert!(book.best_ask_price().is_none());
    }

    #[test]
    fn self_trade_prevention_cancels_front_maker() {
        let owner = 61;
        let mut book = OrderBook::new();

        book.add_order_to_book(limit_order(owner, Side::Ask, dec!(10.25), dec!(2.000)));
        let incoming = limit_order(owner, Side::Bid, dec!(10.25), dec!(1.000));
        let (trades, cancelled) = book.process_limit_order(incoming);

        assert!(trades.is_empty());
        assert_eq!(cancelled.len(), 1);
        assert_eq!(cancelled[0].owner_id, owner);
        assert_eq!(book.arena_len(), 1);
    }

    #[test]
    fn deep_cancel_storm_returns_arena_to_baseline() {
        let mut book = OrderBook::new();
        let mut ids = Vec::new();
        for owner in 100..300 {
            let order = limit_order(owner, Side::Bid, dec!(1.00), dec!(1.000));
            ids.push(order.id);
            book.add_order_to_book(order);
        }
        assert_eq!(book.arena_len(), ids.len());

        for order_id in ids {
            let _ = book.cancel_order(order_id);
        }

        assert_eq!(book.arena_len(), 0);
        assert_eq!(book.active_order_count(), 0);
        assert!(book.best_bid_price().is_none());
    }
}
