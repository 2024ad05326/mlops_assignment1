import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_clean_data(path: str = 'data/raw/heart_disease_raw.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    df['target'] = (df['num'] > 0).astype(int)
    return df


def build_preprocessor():
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])


def prepare_splits(path: str = 'data/raw/heart_disease_raw.csv', test_size: float = 0.2, random_state: int = 42):
    df = load_and_clean_data(path)
    X = df[FEATURE_COLUMNS]
    y = df['target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_preprocessor(preprocessor, path: str = 'models/preprocessor.pkl'):
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, path)
