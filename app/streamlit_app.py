"""
Cancer Subtype Clustering — Interactive Dashboard
==================================================
A Streamlit application for exploring, clustering, and evaluating
gene expression data from the UCI TCGA RNA-Seq dataset.

Pages
-----
1. Dataset Overview & EDA
    High-level dataset statistics, label distribution, expression value
    distributions, and per-gene variance analysis.

2. PCA Explorer
    Explained-variance curves for the fitted PCA, and a 2D scatter of
    the first two principal components coloured by true label or cluster.

3. Clustering Playground
    Interactive controls to pick an algorithm (K-Means or Agglomerative)
    and number of clusters k. Results are shown as UMAP or t-SNE scatter
    plots alongside a contingency matrix vs the true cancer subtypes.
    Silhouette, ARI, and NMI are computed on the fly.

4. Evaluation Dashboard
    Side-by-side comparison of all pre-computed clustering runs using
    Silhouette, ARI, and NMI metrics, plus projection plots and the
    contingency matrix for the best-performing algorithm.

5. Predict Your Sample
    Upload a CSV of gene expression values (one row per sample, one
    column per gene). The app applies the same preprocessing pipeline
    (log1p → StandardScaler → PCA) and assigns each sample to a cluster
    using the pre-trained K-Means model. Results are shown on the UMAP
    embedding and can be downloaded as a CSV.

Usage
-----
Run from the repository root:
    streamlit run app/streamlit_app.py

All data and model files are loaded relative to the repository root, so
the working directory must be the project root when launching.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# ---------------------------------------------------------------------------
# Directory paths — resolved relative to this file so they work regardless
# of the working directory (local or Streamlit Community Cloud).
# ---------------------------------------------------------------------------
_ROOT       = Path(__file__).parent.parent
DATA_DIR    = _ROOT / "data"
MODELS_DIR  = _ROOT / "models"
FIGURES_DIR = _ROOT / "reports" / "figures"


def _show_image(path: Path, **kwargs):
    """Display an image if the file exists; show an info box otherwise.

    Uses use_column_width internally for compatibility with all Streamlit versions.
    """
    if path.exists():
        kwargs.pop("use_container_width", None)
        st.image(str(path), use_column_width="always", **kwargs)
    else:
        st.info(f"Figure not found: `{path.name}`. Re-run the EDA notebook to regenerate it.")

# ---------------------------------------------------------------------------
# Data loaders
#
# @st.cache_data   — caches serialisable return values (DataFrames, arrays).
#                    The cache is invalidated when the function arguments or
#                    the function source code change.
# @st.cache_resource — caches non-serialisable objects such as sklearn models
#                      that should be loaded once and shared across sessions.
# ---------------------------------------------------------------------------

@st.cache_data
def load_labels():
    """Load cancer subtype labels from the pre-computed cluster assignments file.

    Labels are stored in cluster_assignments.csv (committed to the repo) so the
    app never needs to download the large raw data files at runtime.

    Returns
    -------
    pd.Series
        Cancer subtype for each sample — one of BRCA, KIRC, COAD, LUAD, PRAD.
        Used only for evaluation; never passed to clustering algorithms.
    """
    return pd.read_csv(DATA_DIR / "cluster_assignments.csv", index_col=0)["true_label"]


@st.cache_data
def load_pca_matrix():
    """Load the PCA-reduced feature matrix produced by the preprocessing notebook.

    Returns
    -------
    pd.DataFrame
        Shape (801, n_components). Each column is one principal component.
        n_components is however many PCs were needed to reach 95% explained variance.
    """
    return pd.read_csv(DATA_DIR / "X_pca.csv", index_col=0)


@st.cache_data
def load_embeddings():
    """Load pre-computed 2D t-SNE and UMAP coordinates.

    Both embeddings were fitted on the PCA-reduced matrix. Loading them
    here avoids re-fitting (which can take 30–60 seconds) on every page visit.

    Returns
    -------
    tsne : pd.DataFrame
        t-SNE 2D coordinates with columns ['x', 'y'].
    umap : pd.DataFrame
        UMAP 2D coordinates with columns ['x', 'y'].
    """
    tsne = pd.read_csv(DATA_DIR / "X_tsne.csv", index_col=0)
    umap = pd.read_csv(DATA_DIR / "X_umap.csv", index_col=0)
    return tsne, umap


@st.cache_data
def load_assignments():
    """Load pre-computed cluster assignments from the clustering notebook.

    Returns
    -------
    pd.DataFrame
        Columns: true_label, kmeans_k5, agg_ward, agg_complete, agg_average.
        One row per sample.
    """
    return pd.read_csv(DATA_DIR / "cluster_assignments.csv", index_col=0)


@st.cache_data
def load_metrics():
    """Load the evaluation metrics table from the visualization notebook.

    Returns
    -------
    pd.DataFrame
        Columns: Algorithm, Silhouette, ARI, NMI. One row per algorithm.
    """
    return pd.read_csv(DATA_DIR / "eval_metrics.csv")


@st.cache_resource
def load_models():
    """Load the four trained sklearn / UMAP model objects.

    Uses @st.cache_resource because joblib-loaded models are not
    serialisable by Streamlit's standard cache.

    Returns
    -------
    scaler : StandardScaler
        Fitted on the training set after log1p transform.
    pca : PCA
        Fitted to retain 95% of explained variance (~100 components).
    km5 : KMeans
        K-Means model fitted at k=5 with n_init=50.
    umap_model : UMAP
        UMAP model fitted on the PCA-reduced training matrix.
        Used to project new (uploaded) samples into the existing 2D layout.
    """
    scaler      = joblib.load(MODELS_DIR / "scaler.joblib")
    pca         = joblib.load(MODELS_DIR / "pca.joblib")
    km5         = joblib.load(MODELS_DIR / "kmeans_k5.joblib")
    umap_model  = joblib.load(MODELS_DIR / "umap.joblib")
    return scaler, pca, km5, umap_model


@st.cache_data
def load_kept_genes():
    """Load the list of genes that survived zero-variance filtering.

    The preprocessing notebook removes genes with zero variance across all
    samples before fitting the scaler and PCA. Any new sample submitted for
    prediction must contain exactly these genes.

    Returns
    -------
    list of str
        Gene identifiers in the order expected by the scaler.
    """
    return pd.read_csv(DATA_DIR / "kept_genes.csv")["gene"].tolist()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def scatter_fig(coords, color_series, title, width=6, height=5):
    """Create a 2D scatter plot coloured by a categorical series.

    Parameters
    ----------
    coords : pd.DataFrame
        Must contain columns 'x' and 'y' (2D embedding coordinates).
    color_series : pd.Series
        Categorical values used to colour each point (e.g. cluster labels
        or true subtype labels). Will be re-indexed to match coords.
    title : str
        Plot title.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Re-index to guarantee the boolean mask from (color_series == cat)
    # aligns with the coords DataFrame index.
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
# Page: Dataset Overview & EDA
# ---------------------------------------------------------------------------

def page_overview():
    """Render the dataset overview and exploratory analysis page."""
    st.title("Dataset Overview & EDA")
    st.markdown(
        "High-level summary of the TCGA Gene Expression RNA-Seq dataset. "
        "Plots were generated during the exploratory analysis notebook and are "
        "displayed here as pre-computed figures."
    )

    labels     = load_labels()
    kept_genes = load_kept_genes()
    X_pca_df   = load_pca_matrix()

    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", X_pca_df.shape[0])
    col2.metric("Genes (original)", 20_531)
    col3.metric("Cancer subtypes", labels.nunique())

    # --- Label distribution ---
    st.subheader("Label distribution")
    st.markdown(
        "The five cancer subtypes are reasonably balanced, with BRCA being the most "
        "frequent. Labels are **never** used during clustering — only for evaluation."
    )
    label_counts = labels.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    label_counts.plot(kind="bar", ax=ax, edgecolor="white",
                      color=sns.color_palette("Set2", len(label_counts)))
    ax.set_xlabel("Cancer Subtype"); ax.set_ylabel("Number of samples")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                int(bar.get_height()), ha="center", va="bottom", fontsize=10)
    ax.set_title("Sample Count per Cancer Subtype")
    fig.tight_layout()
    st.pyplot(fig)

    # --- Pre-computed figures from the EDA notebook ---
    st.subheader("Expression value distribution")
    st.markdown(
        "RNA-Seq data is heavily right-skewed and sparse. The left plot shows the "
        "raw distribution; the right shows only non-zero values."
    )
    _show_image(FIGURES_DIR / "expression_distribution.png", use_container_width=True)

    st.subheader("Per-gene variance")
    st.markdown(
        "Many genes have near-zero variance across all samples and are removed before "
        f"PCA. **{len(kept_genes):,}** genes survived the zero-variance filter."
    )
    _show_image(FIGURES_DIR / "gene_variance.png", use_container_width=True)

    st.subheader("Mean expression — top 20 most variable genes per subtype")
    _show_image(FIGURES_DIR / "heatmap_top20_genes.png", use_container_width=True)


# ---------------------------------------------------------------------------
# Page: PCA Explorer
# ---------------------------------------------------------------------------

def page_pca():
    """Render the PCA explorer page."""
    st.title("PCA Explorer")
    st.markdown(
        "Principal Component Analysis (PCA) was used to reduce the ~20,000-gene "
        "feature space to a compact set of components. This page lets you inspect "
        "how much variance each component captures and how the samples separate in "
        "the first two principal components."
    )

    X_pca_df    = load_pca_matrix()
    labels      = load_labels()
    assignments = load_assignments()
    _, pca, *_  = load_models()

    # --- Explained variance ---
    st.subheader("Explained variance")
    st.markdown(
        "The left plot shows the variance captured by each individual component. "
        "The curve drops sharply — most of the variance is concentrated in the first "
        "few dozen components. The right plot shows the cumulative sum; the dashed "
        "lines mark the number of components needed to reach 80%, 90%, and 95%."
    )
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    n_show = min(100, len(pca.explained_variance_ratio_))
    axes[0].plot(range(1, n_show + 1),
                 pca.explained_variance_ratio_[:n_show] * 100,
                 color="steelblue", linewidth=1.5)
    axes[0].set_title("Variance per component (first 100 PCs)")
    axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Variance explained (%)")

    axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100,
                 color="darkorange", linewidth=1.5)
    for thresh, color, label in [
        (0.80, "green", "80%"), (0.90, "blue", "90%"), (0.95, "red", "95%")
    ]:
        n = int(np.searchsorted(cumvar, thresh)) + 1
        axes[1].axhline(thresh * 100, linestyle="--", color=color,
                        linewidth=0.9, label=f"{label} variance @ PC{n}")
    axes[1].set_xlabel("Number of components"); axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative explained variance")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.metric("Components retained (95% threshold)", pca.n_components_)
    col2.metric("Total variance retained",
                f"{pca.explained_variance_ratio_.sum()*100:.2f}%")

    # --- 2D PCA scatter ---
    st.subheader("2D PCA projection")
    st.markdown(
        "The two largest principal components capture the most variance in the data. "
        "Select a colouring below to see how well the first two PCs separate the "
        "samples by true label or by clustering result."
    )
    colour_by = st.selectbox(
        "Colour points by",
        ["True label"] + list(assignments.columns[1:])
    )

    from sklearn.decomposition import PCA as _PCA
    pca2d  = _PCA(n_components=2, random_state=42)
    coords = pd.DataFrame(
        pca2d.fit_transform(X_pca_df.values),
        index=X_pca_df.index, columns=["x", "y"]
    )

    series = labels if colour_by == "True label" else assignments[colour_by]

    fig = scatter_fig(coords, series, f"PCA 2D — {colour_by}")
    fig.axes[0].set_xlabel(f"PC1 ({pca2d.explained_variance_ratio_[0]*100:.1f}% variance)")
    fig.axes[0].set_ylabel(f"PC2 ({pca2d.explained_variance_ratio_[1]*100:.1f}% variance)")
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Page: Clustering Playground
# ---------------------------------------------------------------------------

def page_clustering():
    """Render the interactive clustering playground page."""
    st.title("Clustering Playground")
    st.markdown(
        "Choose an algorithm and number of clusters from the sidebar. The model is "
        "fitted live on the PCA-reduced data, then the resulting clusters are "
        "visualised on the selected 2D projection and compared to the true labels "
        "via a contingency matrix."
    )

    X_pca_df        = load_pca_matrix()
    labels          = load_labels()
    X_tsne, X_umap  = load_embeddings()

    st.sidebar.header("Clustering parameters")
    algo = st.sidebar.selectbox(
        "Algorithm",
        ["K-Means", "Agglomerative (ward)", "Agglomerative (complete)", "Agglomerative (average)"]
    )
    k    = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=5)
    proj = st.sidebar.radio("2D projection", ["UMAP", "t-SNE"])

    X_pca = X_pca_df.values

    with st.spinner("Fitting clustering model …"):
        if algo == "K-Means":
            # n_init=20 runs 20 random initialisations and picks the best result
            # by inertia, reducing sensitivity to the random starting positions.
            model = KMeans(n_clusters=k, n_init=20, random_state=42)
            pred  = model.fit_predict(X_pca)
        else:
            linkage = algo.split("(")[1].rstrip(")")
            model   = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            pred    = model.fit_predict(X_pca)

    sil = silhouette_score(X_pca, pred)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette", f"{sil:.4f}",
                help="Cluster compactness vs separation. Range: −1 to +1. Higher is better.")
    col2.metric("ARI", f"{ari:.4f}",
                help="Agreement with true labels, corrected for chance. 1 = perfect.")
    col3.metric("NMI", f"{nmi:.4f}",
                help="Normalised mutual information with true labels. 1 = perfect.")

    coords = (X_umap if proj == "UMAP" else X_tsne).copy()
    coords.index = X_pca_df.index
    pred_series  = pd.Series(pred, index=X_pca_df.index)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Predicted clusters")
        fig = scatter_fig(coords, pred_series, f"{proj} — {algo} (k={k})")
        st.pyplot(fig)
    with col_b:
        st.subheader("True labels")
        fig = scatter_fig(coords, labels, f"{proj} — True Labels")
        st.pyplot(fig)

    # Contingency matrix: rows = predicted cluster, columns = true subtype.
    # Each cell shows how many samples from a given subtype ended up in each cluster.
    st.subheader("Contingency matrix")
    st.markdown(
        "Each cell shows the number of samples with a given true label (column) "
        "assigned to a given cluster (row). A good clustering should have one "
        "dominant subtype per cluster."
    )
    ct = pd.crosstab(pred_series.rename("Cluster"), labels.rename("True Label"))
    fig, ax = plt.subplots(figsize=(8, max(3, k * 0.6)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues",
                linewidths=0.4, cbar=False, ax=ax)
    ax.set_title(f"{algo} (k={k}) vs True Labels")
    fig.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Page: Evaluation Dashboard
# ---------------------------------------------------------------------------

def page_evaluation():
    """Render the evaluation metrics dashboard page."""
    st.title("Evaluation Dashboard")
    st.markdown(
        "This page summarises the performance of all four pre-computed clustering "
        "variants across three metrics. Green cells highlight the best value in each "
        "column. Use the projection toggle to visually compare how each algorithm "
        "partitions the data in 2D."
    )

    metrics         = load_metrics()
    assignments     = load_assignments()
    X_pca_df        = load_pca_matrix()
    labels          = load_labels()
    X_tsne, X_umap  = load_embeddings()

    # Metrics table with best-value highlighting
    st.subheader("Metrics comparison")
    styled = (
        metrics.style
        .format({"Silhouette": "{:.4f}", "ARI": "{:.4f}", "NMI": "{:.4f}"})
        .highlight_max(subset=["Silhouette", "ARI", "NMI"], color="#d4edda")
    )
    st.dataframe(styled, use_container_width=True)

    # Side-by-side ARI / NMI bar chart
    st.subheader("ARI & NMI comparison")
    st.markdown(
        "ARI and NMI both measure agreement with the ground-truth labels. "
        "ARI corrects for chance agreement; NMI captures shared information. "
        "Both range from 0 (random) to 1 (perfect)."
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    x     = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, metrics["ARI"], width, label="ARI", color="steelblue")
    ax.bar(x + width / 2, metrics["NMI"], width, label="NMI", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["Algorithm"], rotation=15, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Clustering Evaluation — ARI & NMI vs True Labels")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # Per-algorithm projection grid
    st.subheader("Projections per algorithm")
    proj   = st.radio("Projection", ["UMAP", "t-SNE"], horizontal=True)
    coords = X_umap if proj == "UMAP" else X_tsne

    cluster_cols = [c for c in assignments.columns if c != "true_label"]
    cols = st.columns(len(cluster_cols))
    for col_ui, algo_col in zip(cols, cluster_cols):
        with col_ui:
            fig = scatter_fig(coords, assignments[algo_col],
                              algo_col, width=4, height=4)
            st.pyplot(fig)

    # Contingency matrix for the best algorithm by ARI
    best = metrics.sort_values("ARI", ascending=False).iloc[0]["Algorithm"]
    st.subheader(f"Best algorithm by ARI: `{best}`")
    ct = pd.crosstab(
        assignments[best].rename("Cluster"),
        labels.rename("True Label")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues",
                linewidths=0.4, cbar_kws={"label": "Number of samples"}, ax=ax)
    ax.set_title(f"Contingency Matrix — {best} vs True Labels")
    fig.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Page: Predict Your Sample
# ---------------------------------------------------------------------------

def page_predict():
    """Render the sample prediction page."""
    st.title("Predict Your Sample")
    st.markdown(
        "Upload a CSV file with **one row per sample** and **one column per gene**, "
        "using the same gene identifiers as the training data. "
        "The app will apply the full preprocessing pipeline — log1p transform, "
        "standardisation, and PCA projection — then assign each sample to one of the "
        "five clusters using the pre-trained K-Means model."
    )
    scaler, pca, km5, umap_model = load_models()
    kept_genes  = load_kept_genes()
    labels      = load_labels()
    X_umap      = load_embeddings()[1]

    uploaded = st.file_uploader("Upload gene expression CSV", type=["csv"])

    if uploaded is None:
        st.info(
            "Upload a CSV file to get started. "
            "The file must have one row per sample and one column per gene, "
            "using the same gene identifiers as the training data."
        )
        st.stop()

    sample_df = pd.read_csv(uploaded, index_col=0)

    st.subheader("Input preview")
    st.caption("Showing first 8 genes only.")
    st.dataframe(sample_df.iloc[:, :8], use_container_width=True)

    # Validate that all required genes are present in the uploaded file.
    # Missing genes would make the scaler/PCA produce incorrect projections.
    missing = [g for g in kept_genes if g not in sample_df.columns]
    if missing:
        st.error(
            f"{len(missing)} expected genes are missing from the uploaded file "
            f"(e.g. {missing[:5]}). Make sure the CSV uses the same gene identifiers "
            f"as the training data."
        )
        st.stop()

    # --- Preprocessing pipeline (mirrors the preprocessing notebook exactly) ---
    X_aligned = sample_df[kept_genes]           # select & order genes
    X_log     = np.log1p(X_aligned.values)      # log1p: same transform used in training
    X_scaled  = scaler.transform(X_log)         # standardise using training set statistics
    X_pca_s   = pca.transform(X_scaled)         # project into PCA space
    clusters  = km5.predict(X_pca_s)            # assign to nearest centroid
    X_umap_s  = umap_model.transform(X_pca_s)  # project into existing UMAP layout

    result = pd.DataFrame({
        "Sample"                         : sample_df.index,
        "Predicted cluster (K-Means k=5)": clusters,
    })
    st.subheader("Predictions")
    st.dataframe(result, use_container_width=True)

    # --- Overlay uploaded samples on the training UMAP ---
    st.subheader("Sample position on UMAP embedding")
    st.markdown(
        "Uploaded samples (★) are projected into the same 2D UMAP space as the "
        "801 training samples. Training samples are coloured by their true cancer "
        "subtype label at low opacity for reference."
    )
    fig, ax = plt.subplots(figsize=(8, 6))

    palette    = sns.color_palette("Set2", labels.nunique())
    categories = sorted(labels.unique())
    color_map  = dict(zip(categories, palette))

    for cat in categories:
        mask = labels == cat
        ax.scatter(
            X_umap.loc[mask, "x"], X_umap.loc[mask, "y"],
            s=12, alpha=0.35, edgecolors="none",
            color=color_map[cat], label=cat
        )

    ax.scatter(
        X_umap_s[:, 0], X_umap_s[:, 1],
        s=150, marker="*", color="black", zorder=5,
        edgecolors="white", linewidths=0.5,
        label="Uploaded sample(s)"
    )

    ax.set_title("UMAP — Training data + uploaded samples")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.5, fontsize=8, loc="best", framealpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)

    csv_bytes = result.to_csv(index=False).encode()
    st.download_button(
        label="Download predictions as CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Dataset Overview & EDA" : page_overview,
    "PCA Explorer"           : page_pca,
    "Clustering Playground"  : page_clustering,
    "Evaluation Dashboard"   : page_evaluation,
    "Predict Your Sample"    : page_predict,
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
