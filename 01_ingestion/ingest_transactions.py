import pandas as pd
from pathlib import Path

file_path = '../data/source/transaction_data.xlsx'

def read_data(file_path: str) -> pd.DataFrame:
    """ Load, aggregate, and clean transaction data from Excel. """

    data_dict = pd.read_excel(file_path, sheet_name=None)
    transaction_df = pd.concat(data_dict.values(), ignore_index=False)
    transaction_df = transaction_df.drop_duplicates().dropna(subset=['Description', 'Customer ID'])

    return transaction_df

def batch_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Partition the data by InvoiceDate and write daily CSV data
    to the output directory.
    
    Parameters
    --------------
    df: pd.DataFrame
        Cleaned transaction dataframe containing InvoiceDate column.
    output_path: str
        Base directory for partitioned daily CSVs to be written.

    Output Structure
    --------------
    output_path/
        date=YYYY-MM-DD/
            transactions.csv
    """

    df = df.copy()
    df['InvoiceTimestamp'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceDate'] = df['InvoiceTimestamp'].dt.date

    base_output_dir = Path(output_path)

    for date, day_transactions in df.groupby('InvoiceDate'):
        batch_path = base_output_dir/f'date={date}'
        batch_path.mkdir(parents=True, exist_ok=True)
        day_transactions.to_csv(batch_path/'transactions.csv', index=False)    


if __name__ == "__main__":

    transaction_df = read_data(file_path)
    batch_data(df=transaction_df, output_path='../data/raw/')

