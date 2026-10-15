from decimal import Decimal

import pandas as pd

from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import (
    AccountType,
    AggressorSide,
    BookType,
    Currency,
    CurrencyPair,
    InstrumentId,
    Money,
    OmsType,
    Price,
    Quantity,
    Symbol,
    TradeId,
    TradeTick,
)
from nautilus_trader.trading import Strategy


# 1. Dataset assumptions and instrument configuration.

SOURCE_TIMEZONE = "UTC"  # Timezone of naive values in "datetime".
SIDE_IS_MAKER = False    # Set True if "side" describes the resting order.

instrument = CurrencyPair(
    instrument_id=InstrumentId.from_str("BTCEUR.KRAKEN"),
    raw_symbol=Symbol("BTCEUR"),
    base_currency=Currency.from_str("BTC"),
    quote_currency=Currency.from_str("EUR"),
    price_precision=2,
    size_precision=8,
    price_increment=Price.from_str("0.01"),
    size_increment=Quantity.from_str("0.00000001"),
    ts_event=0,
    ts_init=0,
)


# 2. Load the trades.
# Read numeric fields as strings to avoid introducing float rounding
# while parsing the CSV, and preserve leading zeros in trade IDs.

data = pd.read_csv(
    "market_data_trades.csv",
    dtype={
        "trade_id": "string",
        "pair": "string",
        "side": "string",
        "price": "string",
        "quantity": "string",
        "type": "string",
    },
)

data['trade_id'] = data['trade_id'].fillna(data.index.to_series())
# If you already have a Polars DataFrame instead, replace the CSV read:
# data = polars_df.to_pandas()

required = {
    "trade_id", "datetime", "pair", "side", "price", "quantity", "type",
}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if data.empty:
    raise ValueError("No trades to test")


# 3. Normalize timestamps.

timestamps = pd.to_datetime(data["datetime"], errors="raise")

if timestamps.isna().any():
    raise ValueError("Missing trade timestamps")

if timestamps.dt.tz is None:
    timestamps = timestamps.dt.tz_localize(
        SOURCE_TIMEZONE,
        ambiguous="raise",
        nonexistent="raise",
    )

data["datetime"] = timestamps.dt.tz_convert("UTC")

# Stable ordering preserves source order for equal timestamps.
data = data.sort_values("datetime", kind="stable")


# 4. Validate and convert individual fields.

def make_decimal(value, field):
    if pd.isna(value):
        raise ValueError(f"Missing {field}")

    number = Decimal(str(value))
    if not number.is_finite() or number <= 0:
        raise ValueError(f"{field} must be finite and positive")

    return number


def make_aggressor_side(value):
    if pd.isna(value):
        return AggressorSide.NO_AGGRESSOR

    value = str(value).strip().lower()

    if value in {"buy", "b", "buyer"}:
        return AggressorSide.SELL if SIDE_IS_MAKER else AggressorSide.BUY

    if value in {"sell", "s", "seller"}:
        return AggressorSide.BUY if SIDE_IS_MAKER else AggressorSide.SELL

    if value in {"", "unknown", "none", "no_aggressor"}:
        return AggressorSide.NO_AGGRESSOR

    raise ValueError(f"Unrecognized trade side: {value!r}")


price_increment = Decimal(str(instrument.price_increment))
size_increment = Decimal(str(instrument.size_increment))

trades = []

for row_number, row in data.iterrows():
    try:
        if pd.isna(row.pair):
            raise ValueError("Missing pair")

        if pd.isna(row.trade_id) or not str(row.trade_id).strip():
            raise ValueError("Missing trade ID")

        price = make_decimal(row.price, "price")
        quantity = make_decimal(row.quantity, "quantity")

        # Strict validation: reject values that would need rounding.
        if price % price_increment != 0:
            raise ValueError(
                f"Price {price} is not aligned to {price_increment}"
            )

        if quantity % size_increment != 0:
            raise ValueError(
                f"Quantity {quantity} is not aligned to {size_increment}"
            )

        # Timestamp.value gives Unix nanoseconds.
        ts_event = int(pd.Timestamp(row.datetime).value)
        if not 0 <= ts_event <= 2**64 - 1:
            raise ValueError("Timestamp is outside the supported range")

        trades.append(
            TradeTick(
                instrument_id=instrument.id,
                price=Price.from_str(
                    format(price, f".{instrument.price_precision}f")
                ),
                size=Quantity.from_str(
                    format(quantity, f".{instrument.size_precision}f")
                ),
                aggressor_side=make_aggressor_side(row.side),
                trade_id=TradeId.from_str(str(row.trade_id).strip()),
                ts_event=ts_event,
                # No receipt timestamp is available in this schema.
                ts_init=ts_event,
            )
        )

    except Exception as exc:
        raise ValueError(
            f"Invalid trade at sorted row {row_number}: {exc}"
        ) from exc


# 5. Subscribe a strategy to trades.

class InspectTrades(Strategy):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    
    def __init__(self, *, instrument_id, **_kwargs):
        super().__init__()
        self.instrument_id = instrument_id
        self.received = 0
        self.last_price = None
        self.last_quantity = None
        self.last_trade_id = None

    def on_start(self):
        self.subscribe_trades(instrument_id=self.instrument_id)

    def on_trade(self, trade: TradeTick):
        self.received += 1
        self.last_price = trade.price
        self.last_quantity = trade.size
        self.last_trade_id = trade.trade_id


# 6. Load and replay the trades.

engine = BacktestEngine(BacktestEngineConfig())

try:
    strategy = InspectTrades(instrument_id=instrument.id)

    engine.add_venue(
        venue=instrument.id.venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money.from_str("100_000 EUR")],
        book_type=BookType.L1_MBP,
        bar_execution=False,
        fee_model=MakerTakerFeeModel(
            maker_rate=Decimal("0"),
            taker_rate=Decimal("0"),
        ),
    )

    engine.add_instrument(instrument)
    engine.add_data(trades)
    engine.add_strategy(strategy)
    engine.run()

    assert strategy.received == len(trades), (
        f"Expected {len(trades)} trades, "
        f"received {strategy.received}"
    )

    print(f"Passed: received all {strategy.received} trades\n")
    print(f"Last price: {strategy.last_price}\n")
    print(f"Last quantity: {strategy.last_quantity}\n")
    print(f"Last trade ID: {strategy.last_trade_id}\n")

finally:
    engine.dispose()