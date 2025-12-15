# 📊 Clustering: Concepts, Algorithms, and Evaluation

This repository is designed for **educational and research purposes**, providing a comprehensive introduction to **clustering and unsupervised learning**. It combines **theoretical explanations**, **practical intuition**, and **Python (scikit-learn) examples** to help learners understand clustering from fundamentals to advanced topics.

---

## 1. Supervised vs Unsupervised Learning

### Supervised Learning

Supervised learning works with **labeled data**, where each input has a known output. The model learns a mapping between inputs and outputs.

**Common tasks:**

* **Classification** – predicting discrete labels (spam vs not spam)
* **Regression** – predicting continuous values (house prices)

### Unsupervised Learning

Unsupervised learning works with **unlabeled data**. The goal is to discover hidden patterns or structures in the data.

**Common tasks:**

* **Clustering** – grouping similar data points
* Dimensionality reduction
* Association rule mining

### Classification vs Clustering

| Classification         | Clustering               |
| ---------------------- | ------------------------ |
| Requires labeled data  | No labels required       |
| Predicts known classes | Discovers natural groups |
| Supervised             | Unsupervised             |

Example:

* Classification: Predict if an email is spam
* Clustering: Group customers based on buying behavior

---

## 2. Distance-Based Thinking in Clustering

> “That’s all about distance comparison”

Clustering algorithms rely heavily on **distance and similarity measures**.

### Pairwise Distances

Pairwise distance computes the distance between **every pair of points** in the dataset. These distances form the backbone of clustering algorithms.

### Distance Measures

#### Euclidean Distance (L2)

* Straight-line distance
* Sensitive to scale

```math
 d(x,y) = \sqrt{\sum (x_i - y_i)^2}
```

#### Manhattan Distance (L1)

* Sum of absolute differences
* Useful in grid-like data

```math
 d(x,y) = \sum |x_i - y_i|
```

#### Mahalanobis Distance

* Accounts for variance and correlation
* Scale-invariant
* Useful when features are correlated

```math
 d(x,y) = \sqrt{(x-y)^T \Sigma^{-1} (x-y)}
```

📺 **Video Resource**: Mahalanobis Distance Explained

---

## Data Normalization for Clustering

Distance-based algorithms are sensitive to feature scales.

### Why Normalization Matters

If one feature has a larger range, it dominates the distance calculation.

### Standardization (Z-score Scaling)

```text
x_scaled = (x - mean) / std
```

### Effect on K-Means

* Without scaling → incorrect clusters
* With scaling → balanced contribution of features

---

## Linkage in Hierarchical Clustering (Simple Explanation)

Linkage defines **how distance between clusters is calculated**:

| Linkage Type | Meaning            |
| ------------ | ------------------ |
| Single       | Closest points     |
| Complete     | Farthest points    |
| Average      | Mean distance      |
| Ward         | Minimizes variance |

Implemented using **scipy** and **sklearn**.

---

## 3. That’s All About Distributions

### Maximum Likelihood Estimation (MLE)

MLE estimates model parameters that make observed data **most probable**.

Steps:

1. Assume a probability distribution
2. Write likelihood function
3. Maximize likelihood (or log-likelihood)

MLE is foundational for:

* Gaussian Mixture Models
* Probabilistic clustering

---

## 4. Clustering Models (scikit-learn)

### K-Means Clustering

**Concept**:

* Choose K centroids
* Assign points to nearest centroid
* Update centroids iteratively

**Variants**:

* `KMeans`
* `MiniBatchKMeans` (faster for large data)

📺 Resources:

* StatQuest: K-Means
* K-Means & Hierarchical Clustering

#### Example

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
```

### Choosing the Optimal Number of Clusters

* **Elbow Method** – inertia vs K
* **Silhouette Score** – cluster separation quality

```python
from sklearn.metrics import silhouette_score
silhouette_score(X, labels)
```

---

### DBSCAN

**Density-based clustering**:

* Finds dense regions
* Identifies noise/outliers
* No need to specify number of clusters

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)
```

📺 Resources:

* DBSCAN Part 1 & 2

---

### HDBSCAN (Advanced Density Clustering)

* Hierarchical DBSCAN
* Automatically finds optimal clusters
* Handles varying densities

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
labels = clusterer.fit_predict(X)
```

📺 Resource: How HDBSCAN Works

---

### Agglomerative (Hierarchical) Clustering

* Bottom-up clustering
* Builds dendrogram
* Uses linkage methods

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)
```

📺 Resources:

* Types of Linkages
* Hierarchical Clustering Explained

---

## 5. Clustering Performance Evaluation

### Extrinsic Metrics (Ground Truth Required)

* Rand Index (RI)
* Adjusted Rand Index (ARI)
* Mutual Information (MI)
* V-measure
* Fowlkes–Mallows Index

```python
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y_true, labels)
```

### Intrinsic Metrics (No Labels Required)

* Silhouette Score
* Calinski–Harabasz Index
* Davies–Bouldin Index

```python
from sklearn.metrics import davies_bouldin_score
davies_bouldin_score(X, labels)
```

---

## 6. Additional: Outlier Detection

### Foundations of Outlier Detection

Outliers are rare or abnormal data points that differ significantly from the majority.

📺 Resources:

* Andrew Ng – Anomaly Detection
* Foundations of Outlier Detection

### Methods

#### Local Outlier Factor (LOF)

* Density-based
* Compares local density to neighbors

```python
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor()
outliers = lof.fit_predict(X)
```

#### Isolation Forest

* Randomly isolates anomalies
* Efficient for high dimensions

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest()
outliers = iso.fit_predict(X)
```

#### One-Class SVM

* Learns boundary around normal data

```python
from sklearn.svm import OneClassSVM
ocsvm = OneClassSVM()
outliers = ocsvm.fit_predict(X)
```

---

## Project Purpose

✔ Educational learning
✔ Research experimentation
✔ Interview & academic preparation

This repository emphasizes **conceptual clarity**, **visual intuition**, and **hands-on experimentation**.

---

## Acknowledgements & Learning Resources

* scikit-learn Documentation
* SciPy Documentation
* StatQuest (Josh Starmer)
* Andrew Ng – Machine Learning
* Research papers & open-source tutorials

---

⭐ If you find this repository useful, consider starring it and sharing it with others!
