// Trimmed copy of trade_the_knight/exchange/src/models.rs: order/trade types and fixed-point helpers only.
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::{Decimal, RoundingStrategy};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type PriceTicks = i64;
pub type QuantityLots = i64;
pub type TeamId = i64;
pub type BotId = i64;
pub type UserId = i64;
pub type RecordId = i64;

pub const PRICE_SCALE: i64 = 10_000;
pub const QUANTITY_SCALE: i64 = 1_000;

pub mod string_id {
    use serde::de::{Error, Visitor};
    use serde::{Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S>(value: &i64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<i64, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StringOrNumberVisitor;

        impl<'de> Visitor<'de> for StringOrNumberVisitor {
            type Value = i64;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string or integer id")
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(value)
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                i64::try_from(value).map_err(|_| E::custom("id exceeds i64 range"))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: Error,
            {
                value
                    .parse::<i64>()
                    .map_err(|_| E::custom("invalid id string"))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: Error,
            {
                self.visit_str(&value)
            }
        }

        deserializer.deserialize_any(StringOrNumberVisitor)
    }
}

pub mod option_string_id {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Option<i64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(value) => serializer.serialize_some(&value.to_string()),
            None => serializer.serialize_none(),
        }
    }

    #[allow(dead_code)]
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<serde_json::Value>::deserialize(deserializer)?
            .map(|value| match value {
                serde_json::Value::String(s) => s
                    .parse::<i64>()
                    .map_err(|_| serde::de::Error::custom("invalid id string")),
                serde_json::Value::Number(n) => n
                    .as_i64()
                    .ok_or_else(|| serde::de::Error::custom("invalid integer id")),
                _ => Err(serde::de::Error::custom("invalid optional id value")),
            })
            .transpose()
    }
}

pub fn decimal_to_price_ticks(value: Decimal) -> Result<PriceTicks, String> {
    scaled_decimal_to_i64(value, PRICE_SCALE, 4, "price")
}

pub fn price_ticks_to_decimal(value: PriceTicks) -> Decimal {
    Decimal::from_i128_with_scale(value as i128, 4)
}

pub fn decimal_to_quantity_lots(value: Decimal) -> Result<QuantityLots, String> {
    scaled_decimal_to_i64(value, QUANTITY_SCALE, 3, "quantity")
}

pub fn quantity_lots_to_decimal(value: QuantityLots) -> Decimal {
    Decimal::from_i128_with_scale(value as i128, 3)
}

fn scaled_decimal_to_i64(
    value: Decimal,
    scale: i64,
    fractional_digits: u32,
    label: &str,
) -> Result<i64, String> {
    value
        .round_dp_with_strategy(fractional_digits, RoundingStrategy::MidpointAwayFromZero)
        .checked_mul(Decimal::from(scale))
        .ok_or_else(|| format!("{label} overflow after scaling"))?
        .round_dp_with_strategy(0, RoundingStrategy::MidpointAwayFromZero)
        .to_i64()
        .ok_or_else(|| format!("{label} is outside the supported i64 fixed-point range"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Bid,
    Ask,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Bid => write!(f, "Bid"),
            Side::Ask => write!(f, "Ask"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Limit,
    Market,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: Uuid,
    #[serde(with = "string_id")]
    pub owner_id: BotId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_order_id: Option<String>,
    pub symbol: String,
    pub price: Decimal,
    pub quantity: Decimal,
    #[serde(skip)]
    pub price_ticks: PriceTicks,
    #[serde(skip)]
    pub quantity_lots: QuantityLots,
    pub side: Side,
    pub order_type: OrderType,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlaceOrderRequest {
    pub id: Uuid,
    #[serde(with = "string_id")]
    pub owner_id: BotId,
    #[serde(default)]
    pub client_order_id: Option<String>,
    pub symbol: String,
    pub price: Decimal,
    pub quantity: Decimal,
    pub side: Side,
    pub order_type: OrderType,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub bid_order_id: Uuid,
    pub ask_order_id: Uuid,
    #[serde(with = "string_id")]
    pub buyer_id: BotId,
    #[serde(with = "string_id")]
    pub seller_id: BotId,
    pub symbol: String,
    pub price: Decimal,
    pub quantity: Decimal,
    #[serde(skip)]
    pub price_ticks: PriceTicks,
    #[serde(skip)]
    pub quantity_lots: QuantityLots,
    pub timestamp: i64,
    pub tick: u64,
}

impl Order {
    pub fn new(
        owner_id: BotId,
        symbol: String,
        price: Decimal,
        quantity: Decimal,
        side: Side,
        order_type: OrderType,
    ) -> Self {
        let mut order = Self {
            id: Uuid::new_v4(),
            owner_id,
            client_order_id: None,
            symbol,
            price,
            quantity,
            price_ticks: 0,
            quantity_lots: 0,
            side,
            order_type,
            timestamp: 0,
        };
        order
            .refresh_fixed_fields()
            .expect("valid fixed-point order");
        order
    }

    pub fn refresh_fixed_fields(&mut self) -> Result<(), String> {
        self.price_ticks = decimal_to_price_ticks(self.price)?;
        self.quantity_lots = decimal_to_quantity_lots(self.quantity)?;
        self.quantity = quantity_lots_to_decimal(self.quantity_lots);
        self.price = price_ticks_to_decimal(self.price_ticks);
        Ok(())
    }

    pub fn set_quantity_lots(&mut self, quantity_lots: QuantityLots) {
        self.quantity_lots = quantity_lots;
        self.quantity = quantity_lots_to_decimal(quantity_lots);
    }

    pub fn set_price_ticks(&mut self, price_ticks: PriceTicks) {
        self.price_ticks = price_ticks;
        self.price = price_ticks_to_decimal(price_ticks);
    }
}
impl Trade {
    pub fn refresh_fixed_fields(&mut self) -> Result<(), String> {
        self.price_ticks = decimal_to_price_ticks(self.price)?;
        self.quantity_lots = decimal_to_quantity_lots(self.quantity)?;
        self.quantity = quantity_lots_to_decimal(self.quantity_lots);
        self.price = price_ticks_to_decimal(self.price_ticks);
        Ok(())
    }

    pub fn from_fixed(
        bid_order_id: Uuid,
        ask_order_id: Uuid,
        buyer_id: BotId,
        seller_id: BotId,
        symbol: String,
        price_ticks: PriceTicks,
        quantity_lots: QuantityLots,
        timestamp: i64,
        tick: u64,
    ) -> Self {
        Self {
            bid_order_id,
            ask_order_id,
            buyer_id,
            seller_id,
            symbol,
            price: price_ticks_to_decimal(price_ticks),
            quantity: quantity_lots_to_decimal(quantity_lots),
            price_ticks,
            quantity_lots,
            timestamp,
            tick,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn fixed_point_helpers_snap_to_engine_precision() {
        assert_eq!(decimal_to_price_ticks(dec!(12.34567)).unwrap(), 123_457);
        assert_eq!(decimal_to_quantity_lots(dec!(3.14159)).unwrap(), 3_142);
        assert_eq!(price_ticks_to_decimal(123_457), dec!(12.3457));
        assert_eq!(quantity_lots_to_decimal(3_142), dec!(3.142));
    }

    #[test]
    fn order_refresh_populates_integer_fields() {
        let mut order = Order {
            id: Uuid::nil(),
            owner_id: 0,
            client_order_id: None,
            symbol: "ABC".to_string(),
            price: dec!(99.99994),
            quantity: dec!(1.2349),
            price_ticks: 0,
            quantity_lots: 0,
            side: Side::Bid,
            order_type: OrderType::Limit,
            timestamp: 0,
        };

        order.refresh_fixed_fields().unwrap();

        assert_eq!(order.price_ticks, 999_999);
        assert_eq!(order.quantity_lots, 1_235);
        assert_eq!(order.price, dec!(99.9999));
        assert_eq!(order.quantity, dec!(1.235));
    }
}
