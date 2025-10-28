"""
topics.py — lightweight topic modeling utilities

Two paths:
1) NMF over TF-IDF (fast, pure scikit-learn)
2) KMeans over TF-IDF (clusters as topics)

Return format is friendly for JSON/HTML:
{
  "topics": [
     {"topic_id": 0, "top_terms": [{"term":"...", "weight":0.123}, ...]},
     ...
  ],
  "doc_topics": [ {"doc_id": i, "topic_id": k, "score": s}, ... ]  # soft scores for NMF, hard labels for KMeans
}
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def _top_terms(components: np.ndarray, vocab: List[str], topk: int = 12) -> List[List[Tuple[str, float]]]:
    terms = []
    for row in components:
        idx = np.argsort(row)[::-1][:topk]
        terms.append([(vocab[i], float(row[i])) for i in idx])
    return terms


def topics_nmf(
    texts: List[str],
    *,
    n_topics: int = 10,
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    topk: int = 12,
) -> Dict[str, object]:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words='english'  # Filter common stopwords
    )
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, init="nndsvd", random_state=42, max_iter=400)
    W = nmf.fit_transform(X)  # docs x topics
    H = nmf.components_      # topics x terms
    vocab = vec.get_feature_names_out().tolist()

    topics = []
    for k, terms in enumerate(_top_terms(H, vocab, topk=topk)):
        topics.append({
            "topic_id": int(k),
            "top_terms": [{"term": t, "weight": w} for t, w in terms],
        })

    # Soft assignment: normalize W row-wise
    row_sums = (W.sum(axis=1) + 1e-12)
    P = (W.T / row_sums).T
    doc_topics = []
    for i in range(P.shape[0]):
        k = int(np.argmax(P[i]))
        s = float(P[i, k])
        doc_topics.append({"doc_id": int(i), "topic_id": k, "score": s})

    return {"topics": topics, "doc_topics": doc_topics}


def topics_kmeans(
    texts: List[str],
    *,
    n_clusters: int = 10,
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    topk: int = 12,
) -> Dict[str, object]:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words='english'  # Filter common stopwords
    )
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    vocab = vec.get_feature_names_out().tolist()

    # derive top terms per cluster using centroid weights
    topics = []
    for k in range(n_clusters):
        centroid = km.cluster_centers_[k]
        idx = np.argsort(centroid)[::-1][:topk]
        topics.append({
            "topic_id": int(k),
            "top_terms": [{"term": vocab[i], "weight": float(centroid[i])} for i in idx],
        })

    doc_topics = [{"doc_id": int(i), "topic_id": int(labels[i]), "score": 1.0} for i in range(len(labels))]
    return {"topics": topics, "doc_topics": doc_topics}
