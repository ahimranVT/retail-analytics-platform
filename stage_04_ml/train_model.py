import pandas as pd
from pathlib import Path
import joblib

from stage_02_transformation.build_analytics_tables import load_raw_data
from stage_04_ml.build_features import build_rfm_features

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
def create_churn_labels(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Label customers as churned if no purchase
    occurred within the last 30 days.
    """
    rfm = rfm.copy()
    rfm['churn'] = (rfm['recency'] > 30).astype(int)

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

def save_model(model):
    """
    Save the model to the artifacts directory.
    """
    model_path = ARTIFACT_DIR/'churn_model.pkl'
    joblib.dump(model, model_path)



if __name__ == '__main__':

    # Load raw dataset from partitions
    raw_df = load_raw_data('data/raw')

    # Create rfm features for loaded df
    rfm = build_rfm_features(raw_df)

    # Label customers as churned if their recency is greater than 30 days
    labeled_df = create_churn_labels(rfm)

    # Train XGBoost model on data
    model, auc = train_churn_model(labeled_df)

    # Save the artifact
    save_model(model)
