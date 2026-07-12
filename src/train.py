import os
import joblib
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from xgboost import XGBClassifier

import src.preprocessing as preprocessing


NUMERIC_FEATURES = preprocessing.NUMERIC_FEATURES
CATEGORICAL_FEATURES = preprocessing.CATEGORICAL_FEATURES
FEATURE_COLUMNS = preprocessing.FEATURE_COLUMNS


def get_preprocessor():
    return preprocessing.build_preprocessor()


def log_confusion_matrix(y_true, y_pred, run_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'Confusion Matrix - {run_name}')
    plt.tight_layout()
    path = f'data/plots/cm_{run_name.lower().replace(" ", "_")}.png'
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    mlflow.log_artifact(path)


def log_roc_curve(model, X_test, y_test, run_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=run_name)
    ax.set_title(f'ROC Curve - {run_name}')
    plt.tight_layout()
    path = f'data/plots/roc_{run_name.lower().replace(" ", "_")}.png'
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    mlflow.log_artifact(path)


def evaluate_model(model, X_train, X_test, y_train, y_test, run_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    log_confusion_matrix(y_test, y_pred, run_name)
    log_roc_curve(model, X_test, y_test, run_name)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    mlflow.log_metric('cv_f1_mean', float(cv_scores.mean()))
    mlflow.log_metric('cv_f1_std', float(cv_scores.std()))

    print(f"\n[{run_name}] Metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
    print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return metrics, model


def train_and_track():
    X_train, X_test, y_train, y_test = preprocessing.prepare_splits()
    preprocessor = get_preprocessor()

    mlflow.set_experiment("Heart_Disease_MLOps")

    best_model = None
    best_score = -1.0
    best_name = ""

    models = {
        'LogisticRegression': (
            Pipeline([('preprocessor', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
            {
                'clf__C': [0.01, 0.1, 1.0, 10.0],
                'clf__penalty': ['l2'],
                'clf__solver': ['lbfgs', 'liblinear']
            }
        ),
        'RandomForest': (
            Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
            {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 10, 20],
                'clf__min_samples_split': [2, 5]
            }
        ),
        'XGBoost': (
            Pipeline([('preprocessor', preprocessor), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))]),
            {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [3, 5],
                'clf__learning_rate': [0.05, 0.1]
            }
        )
    }

    for model_name, (pipeline, param_grid) in models.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({
                'model_type': model_name,
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5
            })

            grid = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring='f1', n_jobs=-1, verbose=1)
            grid.fit(X_train, y_train)

            best_estimator = grid.best_estimator_
            mlflow.log_params(grid.best_params_)

            metrics, fitted_model = evaluate_model(best_estimator, X_train, X_test, y_train, y_test, model_name)

            mlflow.log_param('best_cv_score', float(grid.best_score_))

            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_model = fitted_model
                best_name = model_name

    if best_model is not None:
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model_pipeline.pkl")
        preprocessing.save_preprocessor(best_model.named_steps['preprocessor'])
        print(f"\n[SUCCESS] Best model: {best_name} (F1={best_score:.4f}) saved to models/best_model_pipeline.pkl")


if __name__ == "__main__":
    train_and_track()
