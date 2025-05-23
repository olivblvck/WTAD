from cuml.decomposition import PCA as cuPCA
import cupy as cp

def pca_gpu(X, n_components=2):
    """
    PCA na GPU z użyciem cuML
    Źródło: https://docs.rapids.ai/api/cuml/stable/api.html#cuml.decomposition.PCA
    """
    X_gpu = cp.asarray(X)
    model = cuPCA(n_components=n_components)
    X_reduced = model.fit_transform(X_gpu)
    return X_reduced.get()  # konwersja z GPU na CPU
