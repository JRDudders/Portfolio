"""
Graph Analytics Processing Module

Contains business logic for graph analytics tasks:
- Text similarity
- Clustering
- Network graph generation
- Community detection
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import networkx as nx


def compute_similarity_matrix(texts: List[str], method: str = "cosine") -> List[List[float]]:
    """
    Compute similarity matrix for texts

    Args:
        texts: List of text strings
        method: Similarity method ("cosine", "jaccard")

    Returns:
        Similarity matrix as list of lists
    """
    if method == "cosine":
        # Use TF-IDF + cosine similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix.tolist()

    elif method == "jaccard":
        # Simple jaccard similarity
        n = len(texts)
        similarity_matrix = np.zeros((n, n))

        # Tokenize texts
        tokenized = [set(text.lower().split()) for text in texts]

        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    intersection = len(tokenized[i] & tokenized[j])
                    union = len(tokenized[i] | tokenized[j])
                    similarity_matrix[i][j] = intersection / union if union > 0 else 0.0

        return similarity_matrix.tolist()

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def perform_clustering(
    texts: List[str],
    num_clusters: Optional[int] = None,
    method: str = "kmeans"
) -> Dict[str, Any]:
    """
    Cluster texts using specified method

    Args:
        texts: List of text strings
        num_clusters: Number of clusters (optional for some methods)
        method: Clustering method ("kmeans", "hierarchical", "dbscan")

    Returns:
        Dictionary with clustering results
    """
    # Vectorize texts
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    if method == "kmeans":
        if num_clusters is None:
            num_clusters = min(5, len(texts))

        model = KMeans(n_clusters=num_clusters, random_state=42)
        labels = model.fit_predict(X)

        return {
            "method": "kmeans",
            "num_clusters": num_clusters,
            "labels": labels.tolist(),
            "cluster_sizes": [int(np.sum(labels == i)) for i in range(num_clusters)]
        }

    elif method == "hierarchical":
        if num_clusters is None:
            num_clusters = min(5, len(texts))

        model = AgglomerativeClustering(n_clusters=num_clusters)
        labels = model.fit_predict(X.toarray())

        return {
            "method": "hierarchical",
            "num_clusters": num_clusters,
            "labels": labels.tolist(),
            "cluster_sizes": [int(np.sum(labels == i)) for i in range(num_clusters)]
        }

    elif method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
        labels = model.fit_predict(X.toarray())

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return {
            "method": "dbscan",
            "num_clusters": num_clusters,
            "labels": labels.tolist(),
            "num_noise": int(np.sum(labels == -1)),
            "cluster_sizes": [int(np.sum(labels == i)) for i in set(labels) if i != -1]
        }

    else:
        raise ValueError(f"Unknown clustering method: {method}")


def build_network_graph(texts: List[str], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Build network graph from text similarities

    Args:
        texts: List of text strings
        threshold: Similarity threshold for creating edges

    Returns:
        Dictionary with network graph data
    """
    # Compute similarity matrix
    similarity_matrix = np.array(compute_similarity_matrix(texts, method="cosine"))

    # Create graph
    G = nx.Graph()

    # Add nodes
    for i in range(len(texts)):
        G.add_node(i, text=texts[i][:50])  # Store truncated text

    # Add edges above threshold
    edges = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] >= threshold:
                G.add_edge(i, j, weight=float(similarity_matrix[i][j]))
                edges.append({
                    "source": i,
                    "target": j,
                    "weight": float(similarity_matrix[i][j])
                })

    # Compute network metrics
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "edges": edges,
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G) if G.number_of_edges() > 0 else 0.0,
        "connected_components": nx.number_connected_components(G)
    }


def detect_communities(texts: List[str], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Detect communities in text network

    Args:
        texts: List of text strings
        threshold: Similarity threshold for creating edges

    Returns:
        Dictionary with community detection results
    """
    # Compute similarity matrix
    similarity_matrix = np.array(compute_similarity_matrix(texts, method="cosine"))

    # Create graph
    G = nx.Graph()

    # Add nodes
    for i in range(len(texts)):
        G.add_node(i)

    # Add edges above threshold
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] >= threshold:
                G.add_edge(i, j, weight=float(similarity_matrix[i][j]))

    # Detect communities using Louvain algorithm (via greedy modularity)
    from networkx.algorithms import community

    if G.number_of_edges() == 0:
        # No edges - each node is its own community
        communities_list = [[i] for i in range(len(texts))]
    else:
        communities = community.greedy_modularity_communities(G)
        communities_list = [list(c) for c in communities]

    return {
        "num_communities": len(communities_list),
        "communities": communities_list,
        "community_sizes": [len(c) for c in communities_list],
        "modularity": community.modularity(G, communities) if G.number_of_edges() > 0 else 0.0
    }
