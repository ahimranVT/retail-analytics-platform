from stage_02_transformation.build_analytics_tables import load_raw_data
from stage_04_ml.build_features import build_rfm_features
from stage_04_ml.train_model import create_churn_labels, train_churn_model, save_model

import pandas as pd
import joblib

source_filepath = 'data/sample/sample_data_raw.csv'

def test_training_pipeline_runs():

    raw_df = pd.read_csv(source_filepath)
    rfm, snapshot_date = build_rfm_features(raw_df, 90)
    labeled_df = create_churn_labels(rfm, raw_df, snapshot_date)

    model, auc = train_churn_model(labeled_df)

    assert model is not None

def test_auc_above_random():

    raw_df = pd.read_csv(source_filepath)
    rfm, snapshot_date = build_rfm_features(raw_df, 90)
    labeled_df = create_churn_labels(rfm, raw_df, snapshot_date)

    model, auc = train_churn_model(labeled_df)

    assert auc > 0.55

def test_prediction_range():
    
    raw_df = pd.read_csv(source_filepath)
    rfm, snapshot_date = build_rfm_features(raw_df, 90)
    labeled_df = create_churn_labels(rfm, raw_df, snapshot_date)

    model, auc = train_churn_model(labeled_df)

    preds = model.predict_proba(labeled_df[['recency', 'frequency', 'monetary']])[:,1]

    assert preds.min() >=0
    assert preds.max() <=1    

def test_churn_labels_exist():

    raw_df = pd.read_csv(source_filepath)
    rfm, snapshot_date = build_rfm_features(raw_df, 90)
    labeled_df = create_churn_labels(rfm, raw_df, snapshot_date)

    churn_rate = labeled_df['churn'].mean()

    assert churn_rate > 0
    assert churn_rate < 1

def test_model_saves(tmp_path):

    raw_df = pd.read_csv(source_filepath)
    rfm, snapshot_date = build_rfm_features(raw_df, 90)
    labeled_df = create_churn_labels(rfm, raw_df, snapshot_date)

    model, auc = train_churn_model(labeled_df)

    model_path = tmp_path / 'churn_model.pkl'

    save_model(model, model_path)

    assert model_path.exists()

    model = joblib.load(model_path)

    preds = model.predict_proba(labeled_df[['recency', 'frequency', 'monetary']])[:,1]

    assert preds is not None




