from decimal import Decimal

import pandas as pd

from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import (
    AccountType,
    Bar,
    BarType,
    BookType,
    Currency,
    CurrencyPair,
    InstrumentId,
    Money,
    OmsType,
    Price,
    Quantity,
    Symbol,
)
from nautilus_trader.trading import Strategy


# 1. Dataset assumptions and instrument configuration.

SOURCE_TIMEZONE = "UTC"         # Timezone of naive values in "datetime".
TIMESTAMP_IS_BAR_OPEN = True    # Set False if "datetime" marks the bar close.

# The CSV prices were written from floats, e.g. 90136.70000000001.
# Values within this distance of the price grid are snapped to it;
# anything further away is rejected.
FLOAT_TOLERANCE = Decimal("1e-8")

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

# One-minute bars built from last-trade prices by an external source.
bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
bar_interval_ns = bar_type.spec.get_interval_ns()


# 2. Load the bars.
# Read numeric fields as strings to avoid introducing float rounding
# while parsing the CSV.

data = pd.read_csv(
    "market_data_OHLC.csv",
    dtype={
        "pair": "string",
        "open": "string",
        "high": "string",
        "low": "string",
        "close": "string",
    },
)

required = {"datetime", "pair", "open", "high", "low", "close"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if data.empty:
    raise ValueError("No bars to test")

# The schema has no traded volume ("no" is not a base-currency amount).
# Bars therefore carry zero volume.
zero_volume = Quantity.from_str(format(0, f".{instrument.size_precision}f"))


# 3. Normalize timestamps.

timestamps = pd.to_datetime(data["datetime"], errors="raise")

if timestamps.isna().any():
    raise ValueError("Missing bar timestamps")

if timestamps.dt.tz is None:
    timestamps = timestamps.dt.tz_localize(
        SOURCE_TIMEZONE,
        ambiguous="raise",
        nonexistent="raise",
    )

data["datetime"] = timestamps.dt.tz_convert("UTC")

duplicates = data.duplicated(subset=["datetime"], keep=False)
if duplicates.any():
    raise ValueError(f"{duplicates.sum()} bars share a timestamp")

data = data.sort_values("datetime", kind="stable")


# 4. Validate and convert individual fields.

def make_decimal(value, field):
    if pd.isna(value):
        raise ValueError(f"Missing {field}")

    number = Decimal(str(value))
    if not number.is_finite() or number <= 0:
        raise ValueError(f"{field} must be finite and positive")

    return number


price_increment = Decimal(str(instrument.price_increment))


def make_price(value, field):
    price = make_decimal(value, field)

    # Snap float noise onto the grid, but reject genuine misalignment.
    snapped = (price / price_increment).to_integral_value() * price_increment
    if abs(price - snapped) > FLOAT_TOLERANCE:
        raise ValueError(f"{field} {price} is not aligned to {price_increment}")

    return Price.from_str(format(snapped, f".{instrument.price_precision}f"))


expected_pair = instrument.raw_symbol.value

bars = []

for row_number, row in enumerate(data.itertuples(index=False)):
    try:
        if pd.isna(row.pair):
            raise ValueError("Missing pair")

        pair = str(row.pair).replace("-", "").replace("/", "").upper()
        if pair != expected_pair:
            raise ValueError(f"Pair {row.pair!r} does not match {expected_pair}")

        open_price = make_price(row.open, "open")
        high_price = make_price(row.high, "high")
        low_price = make_price(row.low, "low")
        close_price = make_price(row.close, "close")

        if not low_price <= min(open_price, close_price):
            raise ValueError("Low exceeds open or close")

        if not high_price >= max(open_price, close_price):
            raise ValueError("High is below open or close")

        # Timestamp.value gives Unix nanoseconds.
        # Nautilus stamps time bars at their close, so a bar is only
        # visible to strategies once its interval has finished.
        ts_event = int(pd.Timestamp(row.datetime).value)
        if TIMESTAMP_IS_BAR_OPEN:
            ts_event += bar_interval_ns

        if not 0 <= ts_event <= 2**64 - 1:
            raise ValueError("Timestamp is outside the supported range")

        bars.append(
            Bar(
                bar_type=bar_type,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=zero_volume,
                ts_event=ts_event,
                # No receipt timestamp is available in this schema.
                ts_init=ts_event,
            )
        )

    except Exception as exc:
        raise ValueError(
            f"Invalid bar at sorted row {row_number}: {exc}"
        ) from exc


# 5. Subscribe a strategy to bars.

class InspectBars(Strategy):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *, bar_type, **_kwargs):
        super().__init__()
        self.bar_type = bar_type
        self.received = 0
        self.last_bar = None

    def on_start(self):
        self.subscribe_bars(bar_type=self.bar_type)

    def on_bar(self, bar: Bar):
        self.received += 1
        self.last_bar = bar


# 6. Load and replay the bars.

engine = BacktestEngine(BacktestEngineConfig())

try:
    strategy = InspectBars(bar_type=bar_type)

    engine.add_venue(
        venue=instrument.id.venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money.from_str("100_000 EUR")],
        book_type=BookType.L1_MBP,
        # Bars are the only market data, so let them drive the
        # simulated book.
        bar_execution=True,
        fee_model=MakerTakerFeeModel(
            maker_rate=Decimal("0"),
            taker_rate=Decimal("0"),
        ),
    )

    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(strategy)
    engine.run()

    assert strategy.received == len(bars), (
        f"Expected {len(bars)} bars, "
        f"received {strategy.received}"
    )

    print(f"Passed: received all {strategy.received} bars\n")
    print(f"Last bar: {strategy.last_bar}\n")

finally:
    engine.dispose()
