"""
Graph Analytics Processing Module

Contains business logic for graph analytics tasks by wrapping the existing graph_tasks.py module.
"""

from typing import Dict, Any, Optional, List
import pandas as pd

# Import the actual graph analytics implementation
from graph_tasks import (
    load_graph_from_bytes,
    load_ego_network_from_files,
    run_graph_metrics,
    degrees,
    pagerank,
    bfs,
    triangles_undirected,
    eigenvector_centrality,
    betweenness_centrality,
    merge_node_attributes,
    GraphData
)


def load_graph(
    edges_bytes: bytes,
    edges_kind: str = "csv",
    nodes_bytes: Optional[bytes] = None,
    nodes_kind: Optional[str] = None
) -> GraphData:
    """
    Load graph from edge list with optional node attributes

    Args:
        edges_bytes: Edge list file bytes
        edges_kind: File format ("csv", "json", "edge")
        nodes_bytes: Optional node attributes file bytes
        nodes_kind: Optional node file format

    Returns:
        GraphData object
    """
    return load_graph_from_bytes(
        edges_bytes,
        kind=edges_kind,
        nodes_bytes=nodes_bytes,
        nodes_kind=nodes_kind
    )


def load_ego_network(files_dict: Dict[str, bytes], ego_id: Optional[str] = None) -> GraphData:
    """
    Load ego network from multiple files

    Args:
        files_dict: Dictionary mapping file type to bytes (edges, circles, feat, etc.)
        ego_id: Optional ego node ID

    Returns:
        GraphData object with ego network components
    """
    return load_ego_network_from_files(files_dict, ego_id=ego_id)


def compute_metrics(
    graph: GraphData,
    tasks: List[str],
    pagerank_alpha: float = 0.85,
    pagerank_iters: int = 40,
    pagerank_tol: float = 1e-6,
    bfs_source: Optional[str] = None,
    triangles_limit: int = 20000
) -> Dict[str, Any]:
    """
    Compute multiple graph metrics in one call

    Args:
        graph: GraphData object
        tasks: List of tasks ("degrees", "pagerank", "bfs", "triangles")
        pagerank_alpha: PageRank damping factor
        pagerank_iters: PageRank iterations
        pagerank_tol: PageRank tolerance
        bfs_source: Source node for BFS
        triangles_limit: Max nodes for triangle counting

    Returns:
        Dictionary with metric results
    """
    return run_graph_metrics(
        graph,
        tasks=tasks,
        pagerank_alpha=pagerank_alpha,
        pagerank_iters=pagerank_iters,
        pagerank_tol=pagerank_tol,
        bfs_source=bfs_source,
        triangles_limit=triangles_limit
    )


def compute_degrees(graph: GraphData, include_node_attrs: bool = True) -> Dict[str, Any]:
    """
    Compute node degrees

    Args:
        graph: GraphData object
        include_node_attrs: Whether to include node attributes in result

    Returns:
        Dictionary with degree results
    """
    df = degrees(graph)

    if include_node_attrs:
        df = merge_node_attributes(df, graph)

    return {
        "n_nodes": graph.n,
        "degrees": df.to_dict(orient="records"),
        "top_degree": df.iloc[0].to_dict() if len(df) > 0 else None
    }


def compute_pagerank(
    graph: GraphData,
    alpha: float = 0.85,
    iters: int = 40,
    tol: float = 1e-6,
    include_node_attrs: bool = True
) -> Dict[str, Any]:
    """
    Compute PageRank scores

    Args:
        graph: GraphData object
        alpha: Damping factor
        iters: Max iterations
        tol: Convergence tolerance
        include_node_attrs: Whether to include node attributes in result

    Returns:
        Dictionary with PageRank results
    """
    df = pagerank(graph, alpha=alpha, iters=iters, tol=tol)

    if include_node_attrs:
        df = merge_node_attributes(df, graph)

    return {
        "n_nodes": graph.n,
        "pagerank": df.to_dict(orient="records"),
        "top_node": df.iloc[0].to_dict() if len(df) > 0 else None
    }


def compute_bfs(
    graph: GraphData,
    source_node: str,
    include_node_attrs: bool = True
) -> Dict[str, Any]:
    """
    Compute BFS distances from source node

    Args:
        graph: GraphData object
        source_node: Source node ID
        include_node_attrs: Whether to include node attributes in result

    Returns:
        Dictionary with BFS results
    """
    df = bfs(graph, source_node)

    if include_node_attrs:
        df = merge_node_attributes(df, graph)

    # Count reachable nodes
    reachable = int((df['distance'] >= 0).sum())

    return {
        "source": source_node,
        "n_nodes": graph.n,
        "reachable": reachable,
        "unreachable": graph.n - reachable,
        "distances": df.to_dict(orient="records")
    }


def compute_triangles(graph: GraphData, max_nodes: int = 20000) -> Dict[str, Any]:
    """
    Count triangles in undirected graph

    Args:
        graph: GraphData object
        max_nodes: Skip if graph has more nodes than this

    Returns:
        Dictionary with triangle count
    """
    result = triangles_undirected(graph, max_nodes=max_nodes)

    return {
        "n_nodes": graph.n,
        "n_edges": int(graph.edges.shape[0]),
        **result
    }


def compute_eigenvector_centrality(
    graph: GraphData,
    max_iter: int = 100,
    tol: float = 1e-6,
    include_node_attrs: bool = True
) -> Dict[str, Any]:
    """
    Compute eigenvector centrality

    Args:
        graph: GraphData object
        max_iter: Max iterations
        tol: Convergence tolerance
        include_node_attrs: Whether to include node attributes in result

    Returns:
        Dictionary with eigenvector centrality results
    """
    df = eigenvector_centrality(graph, max_iter=max_iter, tol=tol)

    if include_node_attrs:
        df = merge_node_attributes(df, graph)

    return {
        "n_nodes": graph.n,
        "eigenvector_centrality": df.to_dict(orient="records"),
        "top_node": df.iloc[0].to_dict() if len(df) > 0 else None
    }


def compute_betweenness_centrality(
    graph: GraphData,
    normalized: bool = True,
    include_node_attrs: bool = True
) -> Dict[str, Any]:
    """
    Compute betweenness centrality

    Args:
        graph: GraphData object
        normalized: Whether to normalize scores
        include_node_attrs: Whether to include node attributes in result

    Returns:
        Dictionary with betweenness centrality results
    """
    df = betweenness_centrality(graph, normalized=normalized)

    if include_node_attrs:
        df = merge_node_attributes(df, graph)

    return {
        "n_nodes": graph.n,
        "betweenness_centrality": df.to_dict(orient="records"),
        "top_node": df.iloc[0].to_dict() if len(df) > 0 else None
    }


def compute_all_centralities(graph: GraphData) -> Dict[str, Dict[str, float]]:
    """
    Compute all centrality measures for ego network visualization.

    Args:
        graph: GraphData object

    Returns:
        Dictionary with centrality measures:
        {
            "pagerank": {"nodeId": score, ...},
            "betweenness": {"nodeId": score, ...},
            "eigenvector": {"nodeId": score, ...},
            "degree": {"nodeId": score, ...}
        }
    """
    result = {}

    # Compute PageRank
    try:
        pr_df = pagerank(graph, alpha=0.85, iters=40, tol=1e-6)
        result["pagerank"] = dict(zip(pr_df['node'], pr_df['pr']))
    except Exception as e:
        print(f"[graph_processor] PageRank computation failed: {e}")
        result["pagerank"] = {node: 0.0 for node in graph.idx_to_id}

    # Compute Betweenness Centrality
    try:
        bc_df = betweenness_centrality(graph, normalized=True)
        result["betweenness"] = dict(zip(bc_df['node'], bc_df['centrality']))
    except Exception as e:
        print(f"[graph_processor] Betweenness centrality computation failed: {e}")
        result["betweenness"] = {node: 0.0 for node in graph.idx_to_id}

    # Compute Eigenvector Centrality
    try:
        ec_df = eigenvector_centrality(graph, max_iter=100, tol=1e-6)
        result["eigenvector"] = dict(zip(ec_df['node'], ec_df['centrality']))
    except Exception as e:
        print(f"[graph_processor] Eigenvector centrality computation failed: {e}")
        result["eigenvector"] = {node: 0.0 for node in graph.idx_to_id}

    # Compute Degree Centrality
    try:
        deg_df = degrees(graph)
        result["degree"] = dict(zip(deg_df['node'], deg_df['degree']))
    except Exception as e:
        print(f"[graph_processor] Degree computation failed: {e}")
        result["degree"] = {node: 0 for node in graph.idx_to_id}

    return result


def get_graph_info(graph: GraphData, include_centrality: bool = False) -> Dict[str, Any]:
    """
    Get graph information including full nodes/edges for visualization

    Args:
        graph: GraphData object
        include_centrality: If True, compute all centrality measures (for ego networks)

    Returns:
        Dictionary with graph statistics and visualization data
    """
    # Build circles mapping: {circle_name: [node_ids]}
    circles_dict = None
    node_circles_map = {}  # {node_id: [circle_names]}

    if graph.circles is not None:
        circles_dict = {}
        for _, row in graph.circles.iterrows():
            circle = str(row['circle'])
            node = str(row['node'])

            if circle not in circles_dict:
                circles_dict[circle] = []
            circles_dict[circle].append(node)

            if node not in node_circles_map:
                node_circles_map[node] = []
            node_circles_map[node].append(circle)

    # Build nodes array for visualization
    nodes_array = []
    for node_id in graph.idx_to_id:
        node_info = {"id": node_id}

        # Add name attribute if available
        if graph.nodes is not None:
            node_row = graph.nodes[graph.nodes['id'] == node_id]
            if not node_row.empty:
                # Look for 'name' column
                if 'name' in graph.nodes.columns:
                    node_info['name'] = str(node_row.iloc[0]['name'])
                # Add other attributes
                for col in graph.nodes.columns:
                    if col not in ['id', 'idx', 'name']:
                        node_info[col] = node_row.iloc[0][col]

        # Add circles membership
        if node_id in node_circles_map:
            node_info['circles'] = node_circles_map[node_id]

        nodes_array.append(node_info)

    # Build edges array for visualization
    edges_array = []
    for _, row in graph.edges.iterrows():
        src_idx = int(row['src_idx'])
        dst_idx = int(row['dst_idx'])
        edges_array.append({
            "source": graph.idx_to_id[src_idx],
            "target": graph.idx_to_id[dst_idx]
        })

    result = {
        "n_nodes": graph.n,
        "n_edges": int(graph.edges.shape[0]),
        "nodes": nodes_array,
        "edges": edges_array,
        "has_node_attributes": graph.nodes is not None,
        "has_circles": graph.circles is not None,
        "has_features": graph.features is not None,
        "ego_id": graph.ego_id
    }

    # Add circles dictionary if available
    if circles_dict is not None:
        result["circles"] = circles_dict

    # Add feature count if available
    if graph.features is not None:
        feature_cols = [c for c in graph.features.columns if c not in ['id', 'idx']]
        result["feature_count"] = len(feature_cols)

    # Add centrality measures if requested (for ego networks)
    if include_centrality:
        print("[graph_processor] Computing centrality measures for ego network...")
        result["centrality"] = compute_all_centralities(graph)
        print("[graph_processor] Centrality computation complete")

    return result
