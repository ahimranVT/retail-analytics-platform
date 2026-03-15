import pandas as pd
import os
from pathlib import Path
import joblib

from stage_02_transformation.build_analytics_tables import load_raw_data
from stage_04_ml.build_features import build_rfm_features

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
def create_churn_labels(rfm: pd.DataFrame, raw_df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Label customers as churned if they did not make a purchase
    within 30 days of the snapshot date.
    """

    rfm = rfm.copy()
    raw_df = raw_df.copy()

    raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'])
    snapshot_date = pd.to_datetime(snapshot_date)

    future_transactions = raw_df[(raw_df['Quantity'] > 0) &
                                 (raw_df['Price'] > 0) &
                                 (raw_df['InvoiceDate'] > snapshot_date) ]
    
    future_window = future_transactions[
        future_transactions['InvoiceDate'] <= snapshot_date + pd.Timedelta(days=30)
        ]
    
    future_activity = future_window.groupby("Customer ID").size()

    rfm["churn"] = ~rfm["Customer ID"].isin(future_activity.index)
    rfm["churn"] = rfm["churn"].astype(int)


    return rfm

def train_churn_model(df:pd.DataFrame) -> tuple:
    """
    Train churn prediction model using RFM features.
    """
    feature_cols = ['recency', 'frequency', 'monetary']

    X = df[feature_cols]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.2, 
        random_state=2
        )
    
    model = XGBClassifier(
        n_estimators = 200,
        max_depth = 4,
        learning_rate = 0.05,
        colsample_bytree = 1.0,
        subsample = 0.8,
        eval_metric = 'logloss',
        random_state = 2
    )

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)

    print(f'Validation ROC-AUC: {auc:3f}')

    return model, auc 

def save_model(model, path=None):
    """
    Save the model to the artifacts directory.
    """
    if path is None:
        path = ARTIFACT_DIR/'churn_model.pkl'

    joblib.dump(model, path)

if __name__ == '__main__':

    history_window_days = 90

    # Path to transactions CSV downloaded by CI workflow at runtime
    data_path = os.getenv('DATA_PATH', 'data/source/transactions.csv')

    raw_df = pd.read_csv(data_path)

    # Create rfm features for loaded df
    rfm, snapshot_date = build_rfm_features(raw_df, history_window_days)

    # Label customers as churned if their recency is greater than 30 days
    labeled_df = create_churn_labels(rfm, raw_df, snapshot_date)

    # Train XGBoost model on data
    model, auc = train_churn_model(labeled_df)

    # Save the artifact
    save_model(model)
