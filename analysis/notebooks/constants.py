# Constants used across the notebooks for plotting images

DI_DROP_GROUP_NAMES = {
    "0": "Drop A = 0 , B = 0",
    "1": "Drop A = 0 , B = 0.5",
    "2": "Drop A = 0.5 , B = 0",
    "3": "Drop A = 0.5 , B = 0.5",
    "4": "Drop A = 0.75 , Drop B = 0.25",
    "5": "Drop A = 0.25 , Drop B = 0.75",
}
DROP_GROUP_NAME_ORDER = list(DI_DROP_GROUP_NAMES.values())
DI_METRICS_BY_GROUP = {}
DI_METRICS_BY_GROUP["MMD (RBF)"] = [
    {
        "short_name": "mmd_rbf_raw",
        "long_name": "MMD (RBF) - No embedding",
        "embedding": "none",
    },
    {
        "short_name": "mmd_rbf_inception",
        "long_name": "MMD (RBF) - Inception",
        "embedding": "inception",
    },
    {
        "short_name": "mmd_rbf_pca",
        "long_name": "MMD (RBF) - Inception + PCA",
        "embedding": "inception_pca",
    },
    {
        "short_name": "mmd_rbf_umap",
        "long_name": "MMD (RBF) - Inception + UMAP",
        "embedding": "inception_umap",
    },
]
DI_METRICS_BY_GROUP["OTDD"] = [
    {
        "short_name": "otdd_exact_raw",
        "long_name": "OTDD - No embedding",
        "embedding": "none",
    },
    {
        "short_name": "otdd_exact_inception",
        "long_name": "OTDD - Inception",
        "embedding": "inception",
    },
    {
        "short_name": "otdd_exact_pca",
        "long_name": "OTDD - Inception + PCA",
        "embedding": "inception_pca",
    },
    {
        "short_name": "otdd_exact_umap",
        "long_name": "OTDD - Inception + UMAP",
        "embedding": "inception_umap",
    },
]
DI_METRICS_BY_GROUP["PAD (linear)"] = [
    {
        "short_name": "pad_linear_pca",
        "long_name": "PAD (linear) - Inception + PCA",
        "embedding": "inception_pca",
    },
    {
        "short_name": "pad_linear_umap",
        "long_name": "PAD (linear) - Inception + UMAP",
        "embedding": "inception_umap",
    },
]
DI_METRICS_BY_GROUP["PAD (RBF)"] = [
    {
        "short_name": "pad_rbf_pca",
        "long_name": "PAD (RBF) - Inception + PCA",
        "embedding": "inception_pca",
    },
    {
        "short_name": "pad_rbf_umap",
        "long_name": "PAD (RBF) - Inception + UMAP",
        "embedding": "inception_umap",
    },
]
DI_METRICS_BY_GROUP["KL Divergence (approx)"] = [
    {
        "short_name": "kde_pca_kl_approx",
        "long_name": "KL Divergence (approx) - Inception + PCA",
        "embedding": "inception_pca",
    },
    {
        "short_name": "kde_umap_kl_approx",
        "long_name": "KL Divergence (approx) - Inception + UMAP",
        "embedding": "inception_umap",
    },
]
DI_METRICS_BY_GROUP["KDE (L2)"] = [
    {
        "short_name": "kde_gaussian_umap_l2",
        "long_name": "KDE (L2) - Inception + UMAP",
        "embedding": "inception_umap",
    },
]
DI_METRICS_BY_GROUP["KDE (TV)"] = [
    {
        "short_name": "kde_gaussian_umap_tv",
        "long_name": "KDE (TV) - Inception + UMAP",
        "embedding": "inception_umap",
    },
]
DI_EMBEDDINGS = {
    "Inception + UMAP": "inception_umap",
    "Inception + PCA": "inception_pca",
    "Inception": "inception",
    "No embedding": "none",
}
TRANSFORM_ORDER = ["No transform", "Grayscale", "Little blur", "Big blur", "Rotate 180"]
