import os
import pandas as pd
import joblib
from src import preprocessing, train


RAW_DATA_PATH = 'data/raw/heart_disease_raw.csv'


def test_raw_data_exists():
    assert os.path.exists(RAW_DATA_PATH), "Raw dataset not found. Run src/download_data.py first."


def test_raw_data_columns():
    df = pd.read_csv(RAW_DATA_PATH)
    expected_cols = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'}
    assert expected_cols.issubset(set(df.columns)), f"Missing columns: {expected_cols - set(df.columns)}"


def test_load_and_clean_data():
    df = preprocessing.load_and_clean_data(RAW_DATA_PATH)
    assert 'target' in df.columns
    assert set(df['target'].unique()).issubset({0, 1})


def test_preprocessor_transforms():
    X_train, X_test, y_train, y_test = preprocessing.prepare_splits()
    preprocessor = preprocessing.build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    assert X_train_t.shape[0] == X_train.shape[0]
    assert X_train_t.shape[1] > 0


def test_train_produces_model():
    if not os.path.exists('models/best_model_pipeline.pkl'):
        train.train_and_track()
    assert os.path.exists('models/best_model_pipeline.pkl')
    model = joblib.load('models/best_model_pipeline.pkl')
    sample_input = pd.DataFrame([{
        'age': 52.0, 'sex': 1.0, 'cp': 3.0, 'trestbps': 172.0, 'chol': 199.0,
        'fbs': 1.0, 'restecg': 0.0, 'thalach': 162.0, 'exang': 0.0, 'oldpeak': 0.5,
        'slope': 1.0, 'ca': 0.0, 'thal': 7.0
    }])
    preds = model.predict(sample_input)
    assert preds[0] in (0, 1)
