from decimal import Decimal

import pandas as pd

from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import (
    AccountType,
    BookType,
    Currency,
    CurrencyPair,
    InstrumentId,
    Money,
    OmsType,
    Price,
    Quantity,
    QuoteTick,
    Symbol,
)
from nautilus_trader.trading import Strategy


# 1. Dataset assumptions and instrument configuration.

SOURCE_TIMEZONE = "UTC"  # Timezone of naive values in "datetime".

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


# 2. Load the top-of-book rows.
# Read numeric fields as strings to avoid introducing float rounding
# while parsing the CSV.

data = pd.read_csv(
    "market_data_TOB.csv",
    dtype={
        "pair": "string",
        "side": "string",
        "price": "string",
        "quantity": "string",
    },
)

required = {"datetime", "pair", "side", "price", "quantity"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if data.empty:
    raise ValueError("No quotes to test")

# Only the best level forms a quote.
if "level" in data.columns:
    data = data[data["level"] == 0]


# 3. Normalize timestamps.

timestamps = pd.to_datetime(data["datetime"], errors="raise")

if timestamps.isna().any():
    raise ValueError("Missing quote timestamps")

if timestamps.dt.tz is None:
    timestamps = timestamps.dt.tz_localize(
        SOURCE_TIMEZONE,
        ambiguous="raise",
        nonexistent="raise",
    )

data["datetime"] = timestamps.dt.tz_convert("UTC")


# 4. Pivot bid and ask rows into one quote per timestamp.

data["side"] = data["side"].str.strip().str.lower()

unknown_sides = set(data["side"].dropna()) - {"bid", "ask"}
if unknown_sides:
    raise ValueError(f"Unrecognized quote sides: {sorted(unknown_sides)}")

duplicates = data.duplicated(subset=["datetime", "side"], keep=False)
if duplicates.any():
    raise ValueError(
        f"{duplicates.sum()} rows share a timestamp and side"
    )

quotes_data = (
    data.pivot(index="datetime", columns="side", values=["price", "quantity"])
    .sort_index(kind="stable")
)
quotes_data.columns = [f"{side}_{field}" for field, side in quotes_data.columns]
quotes_data = quotes_data.reset_index()


# 5. Validate and convert individual fields.

def make_decimal(value, field):
    if pd.isna(value):
        raise ValueError(f"Missing {field}")

    number = Decimal(str(value))
    if not number.is_finite() or number <= 0:
        raise ValueError(f"{field} must be finite and positive")

    return number


price_increment = Decimal(str(instrument.price_increment))
size_increment = Decimal(str(instrument.size_increment))


def make_price(value, field):
    price = make_decimal(value, field)

    # Strict validation: reject values that would need rounding.
    if price % price_increment != 0:
        raise ValueError(f"{field} {price} is not aligned to {price_increment}")

    return Price.from_str(format(price, f".{instrument.price_precision}f"))


def make_size(value, field):
    size = make_decimal(value, field)

    if size % size_increment != 0:
        raise ValueError(f"{field} {size} is not aligned to {size_increment}")

    return Quantity.from_str(format(size, f".{instrument.size_precision}f"))


quotes = []

for row_number, row in enumerate(quotes_data.itertuples(index=False)):
    try:
        bid_price = make_price(row.bid_price, "bid price")
        ask_price = make_price(row.ask_price, "ask price")

        if bid_price > ask_price:
            raise ValueError("Crossed quote: bid exceeds ask")

        # Timestamp.value gives Unix nanoseconds.
        ts_event = int(pd.Timestamp(row.datetime).value)
        if not 0 <= ts_event <= 2**64 - 1:
            raise ValueError("Timestamp is outside the supported range")

        quotes.append(
            QuoteTick(
                instrument_id=instrument.id,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=make_size(row.bid_quantity, "bid quantity"),
                ask_size=make_size(row.ask_quantity, "ask quantity"),
                ts_event=ts_event,
                # No receipt timestamp is available in this schema.
                ts_init=ts_event,
            )
        )

    except Exception as exc:
        raise ValueError(
            f"Invalid quote at sorted row {row_number}: {exc}"
        ) from exc


# 6. Subscribe a strategy to quotes.

class InspectQuotes(Strategy):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *, instrument_id, **_kwargs):
        super().__init__()
        self.instrument_id = instrument_id
        self.received = 0
        self.last_quote = None
        self.last_spread = None

    def on_start(self):
        self.subscribe_quotes(instrument_id=self.instrument_id)

    def on_quote(self, quote: QuoteTick):
        self.received += 1
        self.last_quote = quote
        self.last_spread = (
            quote.ask_price.as_decimal() - quote.bid_price.as_decimal()
        )


# 7. Load and replay the quotes.

engine = BacktestEngine(BacktestEngineConfig())

try:
    strategy = InspectQuotes(instrument_id=instrument.id)

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
    engine.add_data(quotes)
    engine.add_strategy(strategy)
    engine.run()

    assert strategy.received == len(quotes), (
        f"Expected {len(quotes)} quotes, "
        f"received {strategy.received}"
    )

    print(f"Passed: received all {strategy.received} quotes\n")
    print(f"Last quote: {strategy.last_quote}\n")
    print(f"Last spread: {strategy.last_spread}\n")

finally:
    engine.dispose()
