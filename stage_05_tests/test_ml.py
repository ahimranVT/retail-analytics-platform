from stage_02_transformation.build_analytics_tables import load_raw_data
from stage_04_ml.build_features import build_rfm_features
from stage_04_ml.train_model import create_churn_labels, train_churn_model


def test_training_pipeline_runs():

    raw_df = load_raw_data('data/raw')
    rfm, snapshot_date = build_rfm_features(raw_df, 90)
    labeled_df = create_churn_labels(rfm, raw_df, snapshot_date)

    model, auc = train_churn_model(labeled_df)

    assert model is not None

def test_auc_above_random():

    raw_df = load_raw_data('data/raw')
    rfm, snapshot_date = build_rfm_features(raw_df, 90)
    labeled_df = create_churn_labels(rfm, raw_df, snapshot_date)

    model, auc = train_churn_model(labeled_df)

    assert auc > 0.55

def test_prediction_range():
    
    raw_df = load_raw_data('data/raw')
    rfm, snapshot_date = build_rfm_features(raw_df, 90)
    labeled_df = create_churn_labels(rfm, raw_df, snapshot_date)

    model, auc = train_churn_model(labeled_df)

    preds = model.predict_proba(labeled_df[['recency', 'frequency', 'monetary']])[:,1]

    assert preds.min() >=0
    assert preds.max() <=1    


