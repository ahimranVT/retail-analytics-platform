import pandas as pd
from stage_01_ingestion.ingest_transactions import read_data, batch_data
# from pathlib import Path

def test_batch_partitioning(tmp_path):

    sample_path = 'stage_05_tests/sample_data/sample_transactions.csv'
    df = read_data(sample_path)

    output_dir = tmp_path / 'raw'
    batch_data(df, output_dir)

    created_dirs = list(output_dir.iterdir())

    # Expected number of daily partitions created based on the sample data
    assert len(created_dirs) == 3

def test_row_counts(tmp_path):

    sample_path = 'stage_05_tests/sample_data/sample_transactions.csv'
    df = read_data(sample_path)

    output_dir =  tmp_path / 'raw'
    batch_data(df, output_dir)

    rows_written = 0

    for folder in output_dir.iterdir():

        file = folder / 'transactions.csv'
        day_df = pd.read_csv(file)
        rows_written += len(day_df)

    assert rows_written == len(df)

def test_required_columns_present():

    sample_path = 'stage_05_tests/sample_data/sample_transactions.csv'

    df = read_data(sample_path)

    required_columns = {
        'Customer ID', 
        'Description', 
        'Quantity', 
        'Price'
    }

    assert required_columns.issubset(df.columns)

def test_no_critical_nulls():

    sample_path = 'stage_05_tests/sample_data/sample_transactions.csv'

    df = read_data(sample_path)

    assert df['Customer ID'].notnull().all()
    assert df['Description'].notnull().all()




