import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_and_clean_dataset(path, target_column=None):
    df = pd.read_csv(path)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None

    # Uzupełnianie braków tylko w cechach
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Skalowanie cech
    X = StandardScaler().fit_transform(X)

    return X, y
