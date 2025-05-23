from sklearn.decomposition import PCA

def pca_cpu(X, n_components=2):
    """
    PCA na CPU z użyciem scikit-learn
    Źródło: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    model = PCA(n_components=n_components)
    X_reduced = model.fit_transform(X)
    return X_reduced
