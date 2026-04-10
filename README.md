# Cancer Subtype Clustering

Unsupervised discovery of cancer subtypes from gene expression profiles using the TCGA RNA-Seq dataset.

---

## Project Overview

This project investigates whether unsupervised learning can automatically identify meaningful cancer subtypes by clustering 801 patients based solely on their 20,531-gene expression profiles — without using any label information during training.

**Cancer types (used only for evaluation):** BRCA, KIRC, COAD, LUAD, PRAD

---

## Repository Structure

```
Cancer-Subtype-Clustering/
├── data/               # Raw & processed data (gitignored — see below)
├── notebooks/          # Jupyter notebooks (EDA, modeling, evaluation)
├── src/                # Reusable Python modules
├── app/                # Streamlit application
├── models/             # Saved scaler, PCA, and clustering models (joblib)
├── reports/
│   └── figures/        # Generated plots and visualizations
├── requirements.txt
└── README.md
```

---

## Dataset

**Name:** Gene Expression Cancer RNA-Seq  
**Source:** UCI Machine Learning Repository (TCGA / NIH-NCI)  
**Link:** https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

The dataset (~125 MB) is **not committed to this repository**. To download it:

```bash
# Option 1 — manual download from UCI link above
# Option 2 — using the UCI Python API
pip install ucimlrepo
python - <<'EOF'
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=401)
ds.data.original.to_csv("data/data.csv", index=False)
EOF
```

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd Cancer-Subtype-Clustering

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Pipeline

| Phase | Stage                    | Technique                        |
|-------|--------------------------|----------------------------------|
| 1     | Preprocessing            | Log-transform, StandardScaler    |
| 2     | Dimensionality Reduction | PCA (50–100 components)          |
| 3     | Clustering               | K-Means, Agglomerative           |
| 4     | Visualization            | t-SNE, UMAP                      |
| 5     | Evaluation               | Silhouette Score, ARI, NMI       |

---

## Streamlit App

Run locally from the **repository root**:

```bash
streamlit run app/streamlit_app.py
```

Live demo: *(link added after deployment)*

---

## Deployment (Streamlit Community Cloud)

The app is deployed via [Streamlit Community Cloud](https://share.streamlit.io) (free tier).

### Steps to deploy your own fork

1. **Push the repo to GitHub** — ensure `models/`, `data/` (preprocessed CSVs), and `reports/figures/` are all committed. The raw data files (`data/data.csv`, `data/labels.csv`) stay gitignored.

   > **Note on file sizes:** `models/pca.joblib` is ~85 MB. GitHub accepts files up to 100 MB; the push will succeed with a size warning. If the push is rejected, store the file with [Git LFS](https://git-lfs.com) and add a download step in the app.

2. **Connect on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **New app → Deploy a public app from GitHub**.
   - Set **Main file path** to `app/streamlit_app.py`.
   - Leave **Python version** as detected from `runtime.txt` (`3.11`).
   - Click **Deploy**.

3. **First-run behaviour:** Streamlit Cloud clones the repo, installs `requirements.txt`, and launches the app. All model and data files are loaded from the committed repo — no extra download step is needed.

4. **Update the live URL above** once the deployment is live.
