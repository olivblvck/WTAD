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

import numpy as np

def load_mnist_npz(path="datasets/mnist.npz", sample_size=5000):
    """
    Ładuje dane z MNIST w formacie .npz, zwraca X i y.
    Przycinamy do sample_size dla przyspieszenia PCA/k-means.
    """
    data = np.load(path)
    X = data["x_train"].reshape(-1, 28 * 28)  # Spłaszczenie obrazków 28x28
    y = data["y_train"]

    # Losowa próbka (bo MNIST ma 60 000 przykładów)
    indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]

    # Skalowanie wartości pikseli (0–255) do 0–1
    X_sample = X_sample / 255.0

    return X_sample, y_sample
