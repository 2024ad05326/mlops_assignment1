import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_dataset():
    print("Fetching Heart Disease dataset...")
    heart_disease = fetch_ucirepo(id=45)
    df = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/heart_disease_raw.csv", index=False)
    print("Data saved.")

if __name__ == "__main__":
    download_dataset()
