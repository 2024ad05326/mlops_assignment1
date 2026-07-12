import os
import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def train_and_track():
    df = pd.read_csv('data/raw/heart_disease_raw.csv')
    df['target'] = (df['num'] > 0).astype(int)
    X = df.drop(columns=['num', 'target'])
    y = df['target']
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), 
         ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), 
         ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlflow.set_experiment("Heart_Disease_MLOps")
    
    # Simplified training for brevity; full logic in original files
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
    pipeline.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/best_model_pipeline.pkl")
    print("Model saved.")

if __name__ == "__main__":
    train_and_track()
