# EMBEDDING EXPLORER — Feature Embeddings + Clustering

**MS Computer Vision Coding Project — Rochester Institute of Technology**

**(Poach) Phumapiwat Chanyutthagorn**

---

## Overview

This project explores deep feature extraction using a pretrained ResNet-50 for unsupervised image clustering, high-dimensional embedding visualization, and content-based image retrieval on the CIFAR-10 dataset. A stretch goal demonstrates how a small supervised fine-tuning head improves clustering structure.

The pipeline extracts intermediate CNN representations via forward hooks, clusters them with K-Means, projects them to 2D with t-SNE and UMAP, builds a nearest-neighbor retrieval system, and compares frozen vs. fine-tuned embedding quality.

## Pipeline

| Part | Description |
|------|-------------|
| **Part 0** | Load CIFAR-10 (10,000 image subset) with ResNet and display transforms |
| **Part 1** | Extract features from ResNet-50 `layer2`, `layer3`, `layer4`, and `avgpool` using forward hooks |
| **Part 2** | K-Means clustering (k=10) with Purity, NMI, and Hungarian Accuracy metrics; contact sheet visualization |
| **Part 3** | t-SNE and UMAP dimensionality reduction to 2D for visual cluster analysis |
| **Part 4** | Image retrieval via cosine nearest neighbors with Precision@5 evaluation |
| **Part 5** | *(Stretch)* Supervised linear head fine-tuned on 10% labels; before/after clustering comparison |

## Key Results

### Clustering Metrics (Part 2 — Frozen ResNet-50 avgpool)

| Metric | Score |
|--------|-------|
| Purity | 0.7431 |
| NMI | 0.6025 |
| Hungarian Accuracy | 0.7431 |

### Layer Comparison (Part 1 — NMI by layer depth)

| Layer | Dimension | NMI |
|-------|-----------|-----|
| `layer2` | 512 | 0.1667 |
| `layer3` | 1024 | 0.3429 |
| `layer4` | 2048 | 0.6057 |
| `avgpool` | 2048 | 0.6057 |

Deeper layers produce progressively better clustering, confirming that later ResNet blocks encode more semantically meaningful, class-discriminative features.

### Image Retrieval (Part 4)

| Metric | Score |
|--------|-------|
| Precision@5 | 0.7863 |

### Fine-Tuning Improvement (Part 5 — Frozen vs. Fine-Tuned)

| Metric | Frozen (Part 2) | Fine-Tuned (Part 5) | Improvement |
|--------|-----------------|---------------------|-------------|
| Purity | 0.7431 | 0.8493 | +0.1062 |
| NMI | 0.6025 | 0.7246 | +0.1222 |
| Hungarian Acc | 0.7431 | 0.8493 | +0.1062 |

A small MLP head (2048 → 256 → 10) trained on only 10% of labeled data produces substantial clustering improvement across all metrics.

## Method Details

### Feature Extraction

A pretrained ResNet-50 (ImageNet weights) is used as a frozen feature extractor. Forward hooks capture activations at four depth levels. Convolutional outputs are global-average-pooled to produce a single vector per image. The primary embedding is the 2048-dimensional `avgpool` output, L2-normalized for cosine similarity computation.

### Clustering

K-Means (k=10, 10 random initializations) is applied to L2-normalized embeddings. Evaluation uses three complementary metrics: Cluster Purity measures majority-class dominance per cluster, NMI quantifies information-theoretic agreement, and Hungarian Accuracy finds the optimal one-to-one cluster-to-class mapping via the Hungarian algorithm.

### Visualization

t-SNE (perplexity=30) and UMAP (n_neighbors=15, cosine metric) reduce the 2048-dimensional embeddings to 2D. Both methods reveal clear class-level grouping in the ResNet feature space, with vehicles (airplane, automobile, ship, truck) and animals (bird, cat, deer, dog, frog, horse) forming distinct macro-clusters.

### Image Retrieval

A nearest-neighbor index is built on L2-normalized avgpool embeddings using Euclidean distance (equivalent to cosine similarity on normalized vectors). For each query, the top-5 nearest neighbors are retrieved and evaluated for class agreement.

### Fine-Tuning (Stretch)

A lightweight linear head (`nn.Linear(2048, 256) → ReLU → Dropout(0.3) → nn.Linear(256, 10)`) is trained for 50 epochs on 10% of the data (1,000 samples) with Adam (lr=1e-3) and cross-entropy loss. After training, all 10,000 embeddings are passed through the first layer to produce 256-dimensional specialized features, which are then re-clustered with K-Means.

## Requirements

- Python 3.10+
- PyTorch and torchvision
- scikit-learn
- matplotlib
- umap-learn (optional, for UMAP visualization)
- scipy

Install dependencies:
```
pip install torch torchvision scikit-learn matplotlib umap-learn scipy
```

## Usage

1. Open `EMBEDDING_EXPLORER___Feature_Embeddings___Clustering.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. CIFAR-10 will be auto-downloaded on first run (~170 MB)
4. With CPU: ~15-20 minutes total runtime; with GPU: ~5-10 minutes

## Project Structure

```
├── README.md
├── EMBEDDING_EXPLORER___Feature_Embeddings___Clustering.ipynb
└── outputs/                  (generated at runtime)
    └── [contact sheets, t-SNE/UMAP plots, retrieval panels]
```

## References

1. Xie et al., "Unsupervised Deep Embedding for Clustering Analysis" (DEC), ICML 2016
2. McInnes & Healy, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction," arXiv 2018
3. Caron et al., "Deep Clustering for Unsupervised Learning of Visual Features" (DeepCluster), ECCV 2018
4. Guérin et al., "Combining Pretrained CNN Feature Extractors to Enhance Clustering of Complex Natural Images," Neurocomputing 2021
5. Pedronette et al., "Manifold Information through Neighbor Embedding Projection for Image Retrieval," Pattern Recognition Letters 2024
