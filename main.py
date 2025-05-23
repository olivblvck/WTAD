from src.download_data import download_mnist
download_mnist()  # <-- zadba o to, żeby dane były przed dalszym kodem

from src.preprocessing import load_and_clean_dataset

# Załaduj Iris – przykład
X_iris, y_iris = load_and_clean_dataset("datasets/iris.csv", target_column="species")

# Tutaj dodasz później ładowanie MNIST, PCA itd.
