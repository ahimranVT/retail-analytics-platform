import pandas as pd
from stage_02_transformation.build_analytics_tables import build_active_customers, build_daily_revenue

def test_daily_revenue_schema():
    """Ensure daily revenue table has expected columns and numeric revenue."""

    df = pd.read_csv('stage_05_tests/sample_data/sample_transactions.csv')
    daily_revenue_df = build_daily_revenue(df)

    expected_columns = {'InvoiceDate', 'Revenue'}

    assert expected_columns == set(daily_revenue_df.columns)

    assert daily_revenue_df['Revenue'].dtype in ("float64", "int64")

def test_active_customers_schema():
    """Ensure active customers table has unique Invoice Date keys."""

    df = pd.read_csv('stage_05_tests/sample_data/sample_transactions.csv')
    active_customers_df = build_active_customers(df)

    assert active_customers_df['InvoiceDate'].is_unique

