from src.preprocessing import load_and_clean_dataset
from src.preprocessing import load_mnist_npz
from src.methods_cpu import pca_cpu
from src.methods_gpu import pca_gpu

# Iris
X_iris, _ = load_and_clean_dataset("datasets/iris.csv", target_column="species")

X_cpu = pca_cpu(X_iris, n_components=2)
X_gpu = pca_gpu(X_iris, n_components=2)

print("CPU PCA shape:", X_cpu.shape)
print("GPU PCA shape:", X_gpu.shape)

X_mnist, y_mnist = load_mnist_npz()
print("MNIST shape:", X_mnist.shape)  # powinno być np. (5000, 784)
