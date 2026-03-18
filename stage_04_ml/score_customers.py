from pathlib import Path
import joblib
import pandas as pd

import os

from stage_04_ml.build_features import build_rfm_features

ARTIFACT_DIR = Path('artifacts')
SCORES_DIR = Path('scores')

def load_model(path=None):
    """Load trained model from artifacts directory."""

    if path is None:
        path = ARTIFACT_DIR / 'churn_model.pkl'

    return joblib.load(path)

def score_customers(raw_df: pd.DataFrame, model) -> pd.DataFrame:
    """Generate churn probability scores for all customers."""

    rfm, snapshot_date = build_rfm_features(raw_df, snapshot_window=90)

    feature_cols = ['recency', 'frequency', 'monetary']
    churn_prob = model.predict_proba(rfm[feature_cols])[:,1]

    scores = rfm[['Customer ID']].copy()
    scores['churn_probability'] = churn_prob
    scores = scores.sort_values(by=['churn_probability'], ascending=False).reset_index(drop=True)

    return scores

def save_scores(scores:pd.DataFrame, path=None):
    """Save scores to the scores directory."""

    SCORES_DIR.mkdir(exist_ok=True)

    if path is None:
        path = SCORES_DIR / 'churn_scores.csv'

    scores.to_csv(path, index=False)
    print(f'Scores saved to {path}')

if __name__ == '__main__':

    data_path = os.getenv('DATA_PATH', 'data/source/transactions.csv')

    raw_df = pd.read_csv(data_path)
    model = load_model()
    scores = score_customers(raw_df, model)

    save_scores(scores)


