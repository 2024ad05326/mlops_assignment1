import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda():
    raw_path = "data/raw/heart_disease_raw.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing data source file at {raw_path}. Run download_data.py first.")

    # 1. Load Data
    df = pd.read_csv(raw_path)
    os.makedirs("data/plots", exist_ok=True)
    
    # Map target feature to canonical binary label matching clinical assignment goals
    # 0 = Absence, 1-4 = Presence of Heart Disease
    df['target'] = (df['num'] > 0).astype(int)

    print("--- [EDA Execution] Processing Data Profile Metrics ---")
    print(f"Total Rows (Instances): {df.shape[0]}, Total Features: {df.shape[1]}")
    
    # 2. Missing Value Analysis Matrix
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing_counts, "Percentage (%)": missing_pct})
    print("\nMissing Value Summary Profiles:")
    print(missing_df[missing_df["Missing Count"] > 0])
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title("Missing Value Distribution Heatmap Across Core Patient Attributes")
    plt.tight_layout()
    plt.savefig("data/plots/missing_values_heatmap.png", dpi=150)
    plt.close()

    # 3. Class Target Distribution Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='target', hue='target', palette='Set2', legend=False)
    plt.xticks([0, 1], ['Absence (0)', 'Presence (1)'])
    plt.xlabel("Diagnostic Grouping")
    plt.ylabel("Patient Samples")
    plt.title("Class Balance Evaluation (Target Variable Profile)")
    plt.tight_layout()
    plt.savefig("data/plots/class_distribution.png", dpi=150)
    plt.close()

    # 4. Continuous Feature Matrix Histograms
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(continuous_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data=df, x=col, kde=True, hue='target', multiple='stack', palette='coolwarm')
        plt.title(f"{col.upper()} Distribution Profile")
    plt.tight_layout()
    plt.savefig("data/plots/continuous_distributions.png", dpi=150)
    plt.close()

    # 5. Full Correlation Matrix Heatmap
    # Dropping any absolute string dummy fields like 'name' if present or unprocessed
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="RdBu_r", cbar=True, square=True, annot_kws={"size": 8})
    plt.title("Complete Linear Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig("data/plots/correlation_heatmap.png", dpi=150)
    plt.close()

    print("\n[SUCCESS] Architectural EDA generated. Visual reports saved safely under data/plots/")

if __name__ == "__main__":
    perform_eda()
