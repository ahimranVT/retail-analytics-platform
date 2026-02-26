import pandas as pd
from stage_02_transformation.build_analytics_tables import load_raw_data

def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build customer-level RFM features for churn modeling."""

    df = df.copy()

    df = df[(df["Quantity"] > 0) & 
            (df["Price"] > 0)   
            ]

    df['Revenue'] = df['Quantity'] * df['Price']

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    snapshot_date = df['InvoiceDate'].max()

    rfm = df.groupby('Customer ID').agg(
        last_purchase = ('InvoiceDate', 'max'),
        frequency = ('Invoice', 'nunique'),
        monetary = ('Revenue', 'sum')
    ).reset_index()

    rfm['last_purchase'] = pd.to_datetime(rfm['last_purchase'])

    rfm['recency'] = (snapshot_date - rfm['last_purchase']).dt.days

    rfm = rfm.drop(columns=["last_purchase"])

    return rfm
