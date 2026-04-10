"""
Cancer Subtype Clustering — Streamlit App
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# ---------------------------------------------------------------------------
# Paths (relative to repo root — run with: streamlit run app/streamlit_app.py)
# ---------------------------------------------------------------------------
DATA_DIR   = "data"
MODELS_DIR = "models"
FIG_DIR    = "reports/figures"

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_raw():
    X      = pd.read_csv(f"{DATA_DIR}/data.csv",   index_col=0)
    labels = pd.read_csv(f"{DATA_DIR}/labels.csv", index_col=0)["Class"]
    return X, labels


@st.cache_data
def load_pca_matrix():
    return pd.read_csv(f"{DATA_DIR}/X_pca.csv", index_col=0)


@st.cache_data
def load_embeddings():
    tsne = pd.read_csv(f"{DATA_DIR}/X_tsne.csv", index_col=0)
    umap = pd.read_csv(f"{DATA_DIR}/X_umap.csv", index_col=0)
    return tsne, umap


@st.cache_data
def load_assignments():
    return pd.read_csv(f"{DATA_DIR}/cluster_assignments.csv", index_col=0)


@st.cache_data
def load_metrics():
    return pd.read_csv(f"{DATA_DIR}/eval_metrics.csv")


@st.cache_resource
def load_models():
    scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib")
    pca    = joblib.load(f"{MODELS_DIR}/pca.joblib")
    km5    = joblib.load(f"{MODELS_DIR}/kmeans_k5.joblib")
    umap   = joblib.load(f"{MODELS_DIR}/umap.joblib")
    return scaler, pca, km5, umap


@st.cache_data
def load_kept_genes():
    return pd.read_csv(f"{DATA_DIR}/kept_genes.csv")["gene"].tolist()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SUBTYPE_PALETTE = {
    cls: col
    for cls, col in zip(
        ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"],
        sns.color_palette("Set2", 5),
    )
}


def scatter_fig(coords, color_series, title, width=6, height=5):
    """Return a matplotlib figure for a 2-D scatter coloured by a series."""
    # Re-index color_series to match coords so boolean masks align
    color_series = pd.Series(color_series.values, index=coords.index)
    categories   = sorted(color_series.unique())
    palette      = sns.color_palette("Set2", len(categories))
    color_map    = dict(zip(categories, palette))

    fig, ax = plt.subplots(figsize=(float(width), float(height)))
    for cat in categories:
        mask = color_series == cat
        ax.scatter(
            coords.loc[mask, "x"], coords.loc[mask, "y"],
            s=18, alpha=0.85, edgecolors="none",
            color=color_map[cat], label=str(cat),
        )
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, loc="best", framealpha=0.6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview():
    st.title("Dataset Overview & EDA")

    X, labels = load_raw()

    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", X.shape[0])
    col2.metric("Genes", X.shape[1])
    col3.metric("Cancer subtypes", labels.nunique())

    st.subheader("Sample data")
    st.dataframe(X.iloc[:5, :8], use_container_width=True)

    # Label distribution
    st.subheader("Label distribution")
    label_counts = labels.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    label_counts.plot(kind="bar", ax=ax, edgecolor="white",
                      color=sns.color_palette("Set2", len(label_counts)))
    ax.set_xlabel("Cancer Subtype"); ax.set_ylabel("Samples")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                int(bar.get_height()), ha="center", va="bottom", fontsize=10)
    ax.set_title("Sample Count per Cancer Subtype")
    fig.tight_layout()
    st.pyplot(fig)

    # Expression distribution
    st.subheader("Expression value distribution")
    flat = X.values.flatten()
    rng  = np.random.default_rng(42)
    flat = flat[rng.choice(len(flat), size=min(200_000, len(flat)), replace=False)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].hist(flat, bins=80, color="steelblue", edgecolor="none")
    axes[0].set_title("Raw expression values (sample)")
    axes[0].set_xlabel("Expression")
    nz = flat[flat > 0]
    axes[1].hist(nz, bins=80, color="coral", edgecolor="none")
    axes[1].set_title(f"Non-zero values ({len(nz)/len(flat)*100:.1f}%)")
    axes[1].set_xlabel("Expression")
    fig.tight_layout()
    st.pyplot(fig)

    # Gene variance
    st.subheader("Per-gene variance")
    gene_var = X.var(axis=0)
    st.write(gene_var.describe().rename("Variance").to_frame())

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(gene_var, bins=100, color="mediumseagreen", edgecolor="none")
    ax.set_xlabel("Variance"); ax.set_ylabel("Genes")
    ax.set_title("Per-Gene Variance Distribution")
    fig.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------------------

def page_pca():
    st.title("PCA Explorer")

    X_pca_df = load_pca_matrix()
    _, labels = load_raw()
    assignments = load_assignments()
    scaler, pca, *_ = load_models()

    st.subheader("Explained variance")
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    n_show = min(100, len(pca.explained_variance_ratio_))
    axes[0].plot(range(1, n_show + 1),
                 pca.explained_variance_ratio_[:n_show] * 100,
                 color="steelblue", linewidth=1.5)
    axes[0].set_title("Variance per component (first 100)")
    axes[0].set_xlabel("PC"); axes[0].set_ylabel("Variance (%)")

    axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100,
                 color="darkorange", linewidth=1.5)
    for thresh, color, label in [
        (0.80, "green", "80%"), (0.90, "blue", "90%"), (0.95, "red", "95%")
    ]:
        n = int(np.searchsorted(cumvar, thresh)) + 1
        axes[1].axhline(thresh * 100, linestyle="--", color=color,
                        linewidth=0.9, label=f"{label} @ PC{n}")
    axes[1].set_xlabel("Components"); axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative explained variance")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

    st.metric("Components retained (95% threshold)", pca.n_components_)
    st.metric("Variance retained",
              f"{pca.explained_variance_ratio_.sum()*100:.2f}%")

    # 2-D PCA scatter
    st.subheader("2D PCA projection")
    colour_by = st.selectbox("Colour by", ["True label"] + list(assignments.columns[1:]))

    from sklearn.decomposition import PCA as _PCA
    pca2d  = _PCA(n_components=2, random_state=42)
    coords = pd.DataFrame(
        pca2d.fit_transform(X_pca_df.values),
        index=X_pca_df.index, columns=["x", "y"]
    )

    if colour_by == "True label":
        series = labels
    else:
        series = assignments[colour_by]

    fig = scatter_fig(coords, series, f"PCA 2D — {colour_by}")
    fig.axes[0].set_xlabel(f"PC1 ({pca2d.explained_variance_ratio_[0]*100:.1f}% var)")
    fig.axes[0].set_ylabel(f"PC2 ({pca2d.explained_variance_ratio_[1]*100:.1f}% var)")
    st.pyplot(fig)


# ---------------------------------------------------------------------------

def page_clustering():
    st.title("Clustering Playground")

    X_pca_df    = load_pca_matrix()
    _, labels   = load_raw()
    X_tsne, X_umap = load_embeddings()

    st.sidebar.header("Parameters")
    algo = st.sidebar.selectbox(
        "Algorithm", ["K-Means", "Agglomerative (ward)",
                      "Agglomerative (complete)", "Agglomerative (average)"])
    k    = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)
    proj = st.sidebar.radio("Projection", ["UMAP", "t-SNE"])

    X_pca = X_pca_df.values

    with st.spinner("Fitting clustering …"):
        if algo == "K-Means":
            model  = KMeans(n_clusters=k, n_init=20, random_state=42)
            pred   = model.fit_predict(X_pca)
        else:
            linkage = algo.split("(")[1].rstrip(")")
            model   = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            pred    = model.fit_predict(X_pca)

    sil = silhouette_score(X_pca, pred)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette", f"{sil:.4f}")
    col2.metric("ARI",        f"{ari:.4f}")
    col3.metric("NMI",        f"{nmi:.4f}")

    coords = X_umap if proj == "UMAP" else X_tsne
    coords = coords.copy()
    coords.index = X_pca_df.index

    pred_series = pd.Series(pred, index=X_pca_df.index)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Predicted clusters")
        fig = scatter_fig(coords, pred_series, f"{proj} — {algo} (k={k})")
        st.pyplot(fig)
    with col_b:
        st.subheader("True labels")
        fig = scatter_fig(coords, labels, f"{proj} — True Labels")
        st.pyplot(fig)

    # Contingency matrix
    st.subheader("Contingency matrix")
    ct = pd.crosstab(pred_series.rename("Cluster"), labels.rename("True Label"))
    fig, ax = plt.subplots(figsize=(8, max(3, k * 0.6)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues",
                linewidths=0.4, cbar=False, ax=ax)
    ax.set_title(f"{algo} (k={k}) vs True Labels")
    fig.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------------------

def page_evaluation():
    st.title("Evaluation Metrics Dashboard")

    metrics     = load_metrics()
    assignments = load_assignments()
    X_pca_df    = load_pca_matrix()
    _, labels   = load_raw()
    X_tsne, X_umap = load_embeddings()

    st.subheader("Metrics table")
    styled = (
        metrics.style
        .format({"Silhouette": "{:.4f}", "ARI": "{:.4f}", "NMI": "{:.4f}"})
        .highlight_max(subset=["Silhouette", "ARI", "NMI"], color="#d4edda")
    )
    st.dataframe(styled, use_container_width=True)

    # Bar chart
    st.subheader("ARI & NMI comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    x     = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, metrics["ARI"], width, label="ARI",  color="steelblue")
    ax.bar(x + width / 2, metrics["NMI"], width, label="NMI",  color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["Algorithm"], rotation=15, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Clustering Evaluation vs True Labels")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # Per-algorithm t-SNE / UMAP
    st.subheader("Projections per algorithm")
    proj = st.radio("Projection", ["UMAP", "t-SNE"], horizontal=True)
    coords = X_umap if proj == "UMAP" else X_tsne

    cluster_cols = [c for c in assignments.columns if c != "true_label"]
    n_cols = len(cluster_cols)
    cols   = st.columns(n_cols)
    for col_ui, algo_col in zip(cols, cluster_cols):
        with col_ui:
            fig = scatter_fig(coords, assignments[algo_col],
                              algo_col, width=4, height=4)
            st.pyplot(fig)

    # Best algorithm contingency
    best = metrics.sort_values("ARI", ascending=False).iloc[0]["Algorithm"]
    st.subheader(f"Best algorithm by ARI: `{best}`")
    ct = pd.crosstab(
        assignments[best].rename("Cluster"),
        labels.rename("True Label")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues",
                linewidths=0.4, cbar_kws={"label": "Samples"}, ax=ax)
    ax.set_title(f"Contingency Matrix — {best}")
    fig.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------------------

def page_predict():
    st.title("Predict Your Sample")
    st.write(
        "Upload a CSV with **one row per sample** and **one column per gene** "
        "(same gene names as the training data). "
        "The app will preprocess, project via PCA, and assign a cluster."
    )

    scaler, pca, km5, umap_model = load_models()
    kept_genes = load_kept_genes()
    _, labels  = load_raw()
    X_umap     = load_embeddings()[1]
    assignments = load_assignments()

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("No file uploaded yet. Using three random samples from the training set as a demo.")
        X_raw, _ = load_raw()
        sample_df = X_raw.sample(3, random_state=99)
    else:
        sample_df = pd.read_csv(uploaded, index_col=0)

    st.subheader("Input preview")
    st.dataframe(sample_df.iloc[:, :8], use_container_width=True)

    # Align to kept genes
    missing = [g for g in kept_genes if g not in sample_df.columns]
    if missing:
        st.error(f"{len(missing)} expected genes are missing from the upload "
                 f"(e.g. {missing[:5]}).")
        st.stop()

    X_aligned = sample_df[kept_genes]

    # Preprocess
    X_log    = np.log1p(X_aligned.values)
    X_scaled = scaler.transform(X_log)
    X_pca_s  = pca.transform(X_scaled)
    clusters = km5.predict(X_pca_s)
    X_umap_s = umap_model.transform(X_pca_s)

    result = pd.DataFrame({
        "Sample"         : sample_df.index,
        "Predicted cluster (K-Means k=5)": clusters,
    })
    st.subheader("Predictions")
    st.dataframe(result, use_container_width=True)

    # Plot new points on existing UMAP
    st.subheader("Position on UMAP embedding")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Background: training data coloured by true label
    palette    = sns.color_palette("Set2", labels.nunique())
    categories = sorted(labels.unique())
    color_map  = dict(zip(categories, palette))
    for cat in categories:
        mask = labels == cat
        ax.scatter(X_umap.loc[mask, "x"], X_umap.loc[mask, "y"],
                   s=12, alpha=0.4, edgecolors="none",
                   color=color_map[cat], label=cat)

    # New samples: large markers
    ax.scatter(X_umap_s[:, 0], X_umap_s[:, 1],
               s=120, marker="*", color="black", zorder=5,
               edgecolors="white", linewidths=0.5,
               label="Your sample(s)")

    ax.set_title("UMAP — Training data + uploaded samples")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.5, fontsize=8, loc="best", framealpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)

    # Download results
    csv_bytes = result.to_csv(index=False).encode()
    st.download_button("Download predictions CSV", csv_bytes,
                       file_name="predictions.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Dataset Overview & EDA"  : page_overview,
    "PCA Explorer"            : page_pca,
    "Clustering Playground"   : page_clustering,
    "Evaluation Dashboard"    : page_evaluation,
    "Predict Your Sample"     : page_predict,
}

st.set_page_config(
    page_title="Cancer Subtype Clustering",
    page_icon="🧬",
    layout="wide",
)

with st.sidebar:
    st.title("🧬 Cancer Subtype\nClustering")
    st.markdown("---")
    page = st.radio("Navigate", list(PAGES.keys()))
    st.markdown("---")
    st.caption("UCI Gene Expression RNA-Seq · 801 samples · 20,531 genes")

PAGES[page]()
