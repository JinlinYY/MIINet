import numpy as np
import pandas as pd

def extract_excel_features(filename):
    readbook = pd.read_excel(f'{filename}', engine='openpyxl')
    index = readbook.iloc[:, 0].to_numpy()
    labels = readbook.iloc[:, -1].to_numpy()
    features_df = readbook.iloc[:, 1:-1]
    numeric_features = features_df.select_dtypes(include=[np.number])
    categorical_features = features_df.select_dtypes(exclude=[np.number])
    if not categorical_features.empty:
        categorical_features = pd.get_dummies(categorical_features)
    combined_features = pd.concat([numeric_features, categorical_features], axis=1)
    combined_features = combined_features.to_numpy(dtype=np.float32)

    return index, combined_features, labels

