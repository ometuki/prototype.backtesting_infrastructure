from decimal import Decimal
import pandas as pd
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import (
    AccountType,
    BookOrder,
    BookType,
    Money,
    OmsType,
    OrderBookDepth,
    OrderSide,
    InstrumentId,
    CurrencyPair,
    Symbol,
    Price,
    Quantity,
    Currency,
)
from nautilus_trader.trading import Strategy

from data_processing.market_data_utils import align_order_book_schema

# df = pd.DataFrame([
#     {
#         "timestamp": "2026-10-01T12:00:00.000Z",
#         "received_at": "2026-10-01T12:00:00.002Z",
#         "bids": [
#             ("2500.10", "3.5", 2),
#             ("2500.00", "5.0", 4),
#         ],
#         "asks": [
#             ("2500.20", "2.0", 1),
#             ("2500.30", "4.5", 3),
#         ],
#         "sequence": 42,
#     },
# ])

df = pd.read_csv('market_data_FOB.csv')
# 1. Normalize timestamps.
data = align_order_book_schema(df)


# 2. Define the instrument matching the dataset.
instrument_id = InstrumentId.from_str("BTCEUR.KRAKEN")
instrument = CurrencyPair(
    instrument_id=instrument_id,
    raw_symbol=Symbol("BTCEUR"),
    base_currency=Currency.from_str("BTC"),
    quote_currency=Currency.from_str("EUR"),
    price_precision=2,
    size_precision=6,
    price_increment=Price.from_str("0.01"),
    size_increment=Quantity.from_str("0.000001"),
    ts_event=0,
    ts_init=0,
)
# instrument = TestInstrumentProvider.btceur_kraken()


# 3. Convert one side of a snapshot to BookOrder objects.
def make_side(levels, side):
    normalized = []

    for price, size, count in levels:
        price = Decimal(str(price))
        size = Decimal(str(size))

        if not price.is_finite() or not size.is_finite():
            raise ValueError("Non-finite price or size")
        if price <= 0 or size <= 0:
            raise ValueError("Snapshot levels need positive price and size")
        if int(count) != count or not 0 <= count <= 2**32 - 1:
            raise ValueError("Invalid order count")

        normalized.append((price, size, int(count)))

    # Best bid first; best ask first.
    normalized.sort(
        key=lambda level: level[0],
        reverse=(side == OrderSide.BUY),
    )

    orders = [
        BookOrder(
            side=side,
            price=instrument.make_price(float(price)),
            size=instrument.make_qty(float(size)),
            order_id=0,  # Aggregated L2 levels, not individual orders.
        )
        for price, size, _ in normalized
    ]
    counts = [count for _, _, count in normalized]

    # Precision rounding must not merge distinct levels.
    if len({str(order.price) for order in orders}) != len(orders):
        raise ValueError("Duplicate prices after normalization")
    if any(order.size.as_double() <= 0 for order in orders):
        raise ValueError("Size rounded to zero")

    return orders, counts


snapshots = []

for row in data.itertuples(index=False):
    bids, bid_counts = make_side(row.bids, OrderSide.BUY)
    asks, ask_counts = make_side(row.asks, OrderSide.SELL)

    if bids and asks and bids[0].price > asks[0].price:
        raise ValueError("Crossed snapshot: best bid exceeds best ask")

    snapshots.append(
        OrderBookDepth(
            instrument_id=instrument.id,
            bids=bids,
            asks=asks,
            bid_counts=bid_counts,
            ask_counts=ask_counts,
            flags=0,
            sequence=int(row.sequence),
            ts_event=int(row.timestamp.value),
            ts_init=int(row.received_at.value),
        )
    )

if not snapshots:
    raise ValueError("No snapshots to test")

# 4. Subscribe a strategy to depth snapshots.
class InspectDepth(Strategy):
    def __init__(self, instrument_id):
        super().__init__()
        self.instrument_id = instrument_id
        self.received = 0
        self.last_spread = None

    def on_start(self):
        self.subscribe_book_depth(
            instrument_id=self.instrument_id,
            book_type=BookType.L2_MBP,
            managed=True,
        )

    def on_book_depth(self, depth: OrderBookDepth):
        self.received += 1

        if depth.bids and depth.asks:
            self.last_spread = (
                depth.asks[0].price.as_double()
                - depth.bids[0].price.as_double()
            )

        # Evaluate imbalance, signals, and orders here.


# 5. Configure an L2 venue, load snapshots, and run.
engine = BacktestEngine(BacktestEngineConfig())
strategy = InspectDepth(instrument.id)

try:
    engine.add_venue(
        venue=instrument.id.venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money.from_str("100_000 USDT")],
        book_type=BookType.L2_MBP,
        bar_execution=False,
        fee_model=MakerTakerFeeModel(
            maker_rate=Decimal("0"),
            taker_rate=Decimal("0"),
        ),
    )

    engine.add_instrument(instrument)
    engine.add_data(snapshots)
    engine.add_strategy(strategy)
    engine.run()

    # After engine.run():
    print('--------------------------------------------------------------------------------------------------------------------------')
    print(f"Expected {len(snapshots)} snapshots, received {strategy.received} \n")
    assert strategy.received == len(snapshots), (
        f"Expected {len(snapshots)} snapshots, "
        f"received {strategy.received}\n"
    )
    print(f"Last spread: {strategy.last_spread} \n")
finally:
    engine.dispose()

