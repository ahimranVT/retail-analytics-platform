import pandas as pd
from stage_02_transformation.build_analytics_tables import build_active_customers, build_daily_revenue

def test_revenue_not_negative():
    """Revenue should not be negative across all days."""

    df = pd.read_csv('stage_05_tests/sample_data/sample_transactions.csv')

    revenue_df = build_daily_revenue(df)
    assert not (revenue_df['Revenue'] < 0).any()


def test_revenue_spike_guardrail():
    """Revenue should not exceed 10x previous day's revenue."""

    df = pd.read_csv('stage_05_tests/sample_data/sample_transactions.csv')

    revenue_df = build_daily_revenue(df)
    revenue_df = revenue_df.sort_values(['InvoiceDate'])

    revenue_df['prev_revenue'] = revenue_df['Revenue'].shift(1)
    valid_rows = revenue_df.dropna()

    assert (valid_rows['Revenue'] <= 10 * valid_rows['prev_revenue']).all()

def test_active_customers_not_zero():
    """Active customers should never drop to zero."""

    df = pd.read_csv('stage_05_tests/sample_data/sample_transactions.csv')

    customers_df = build_active_customers(df)

    assert (customers_df['active_customers'] > 0).all()
