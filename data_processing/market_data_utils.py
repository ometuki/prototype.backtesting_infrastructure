import pandas as pd

def align_order_book_schema(input_data: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in input_data.columns and "datetime" not in input_data.columns:
        raise ValueError("Input data must contain a 'timestamp' or 'datetime' column.")
    if "side" not in input_data.columns:
        raise ValueError("Input data must contain a 'side' column.")
    if "price" not in input_data.columns:
        raise ValueError("Input data must contain a 'price' column.")
    if "quantity" not in input_data.columns:
        raise ValueError("Input data must contain a 'quantity' column.")
    
    data = input_data.copy()
    if "timestamp" not in data.columns:
        data = data.rename(columns = {'datetime': 'timestamp'})
    if 'orders' not in data.columns:
        data['orders'] = 0
    data = data.assign(order_tuple=list(zip(data['price'], data['quantity'], data['orders'])))\
      .groupby(['timestamp', 'side'])['order_tuple']\
      .agg(list)\
      .unstack('side')\
      .reindex(columns=['bid', 'ask'])\
      .rename_axis(columns=None)\
      .rename(columns={'bid':'bids','ask':'asks'})\
      .reset_index()

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

    if "received_at" in data.columns:
        data["received_at"] = pd.to_datetime(data["received_at"], utc=True)
    else:
        # Assumes no recorded delivery delay.
        data["received_at"] = data["timestamp"]

    if "sequence" not in data.columns:
        data["sequence"] = 0  # No venue sequence supplied.

    if data.empty:
        raise ValueError("No snapshots")
    if data[["timestamp", "received_at"]].isna().any().any():
        raise ValueError("Missing timestamps")
    if (data["received_at"] < data["timestamp"]).any():
        raise ValueError("Receive time precedes event time")

    # Replay order follows data availability.
    # A stable sort preserves source order for equal timestamps.
    data = data.sort_values("received_at", kind="stable")

    return data