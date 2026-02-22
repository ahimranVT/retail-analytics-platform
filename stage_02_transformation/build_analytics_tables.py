import pandas as pd
from pathlib import Path
from stage_01_ingestion.ingest_transactions import read_data

def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load all daily transaction CSVs from the raw data directory
    and concatenate them into a single dataframe.
    """

    base_path = Path(data_path)

    all_file_paths = base_path.glob('date=*/transactions.csv')

    dfs = [pd.read_csv(file) for file in all_file_paths]

    if not dfs:
        raise ValueError('No transaction files found in raw data directory')
    
    return pd.concat(dfs, ignore_index=True)

def build_daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total revenue by day.
    """
    df = df.copy()
    df['Revenue'] = df['Quantity'] * df['Price']

    return (
        df.groupby('InvoiceDate', as_index=False)['Revenue']
        .sum()
        .sort_values('InvoiceDate')
    )

def build_active_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the number of unique customers by day.
    """
    return (
        df.groupby('InvoiceDate')
        .agg(active_customers = ('Customer ID', 'nunique'))
        .reset_index()
        .sort_values('InvoiceDate')
    )

def apply_transformations(data_path:str, output_path:str) -> None:
    """
    Create sample csvs to test tramsformation functions.
    """

    df = load_raw_data(data_path)

    daily_revenue = build_daily_revenue(df)
    active_customers = build_active_customers(df)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    daily_revenue.to_csv(output_path / 'sample_daily_revenue.csv', index=False)
    active_customers.to_csv(output_path / 'sample_active_customers.csv', index=False)


if __name__ == '__main__':

    apply_transformations('data/raw/',
                          'stage_05_tests/sample_data')