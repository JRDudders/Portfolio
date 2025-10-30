"""
graph_tasks.py — lightweight graph analytics with optional GPU acceleration

What it does
------------
- Loads edge lists from CSV/JSON bytes
- Builds compact integer node index maps
- Computes degrees, PageRank, BFS, and (optionally) triangle counts
- Uses efficient pure-Python/NumPy routines by default
- If cuGraph (RAPIDS) is available, automatically uses GPU acceleration
- If pygraphblas is installed, exposes the adjacency as a GraphBLAS matrix
  (you can extend algorithms to use it later)

Edge CSV schema (auto-detected):
  src, dst[, weight, ts, etype]

Edge JSON schema:
  Either a list of {"src": "...", "dst": "...", ...} objects
  or a dict with key "edges" -> list of such objects.

Edge .edge schema (space/tab-separated):
  src dst [weight]
  # Lines starting with # are comments

Node CSV schema (optional):
  id, [attribute1, attribute2, ...]

Node JSON schema (optional):
  Either a list of {"id": "...", "attr1": "...", ...} objects
  or a dict with key "nodes" -> list of such objects.

Node .node schema (space/tab-separated):
  id [label] [attr1] [attr2] ...
  # Lines starting with # are comments

Ego Network Formats (SNAP-style):
  .edges: nodeID1 nodeID2 (space/tab-separated edge list)
  .circles: circleName node1 node2 node3... (social circles/communities)
  .feat: nodeID feat1 feat2 feat3... (binary node features 0/1)
  .egofeat: feat1 feat2 feat3... (single line, ego node features)
  .featnames: featureID featureName [anonymized] (feature name mappings)
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import typing as T

import numpy as np
import pandas as pd
import networkx as nx

# Optional GraphBLAS
HAS_GB = False
gb = None
try:
    import graphblas as gb  # python-graphblas (preferred on 3.12+)
    HAS_GB = True
except Exception:
    try:
        import pygraphblas as gb  # fallback for older Python
        HAS_GB = True
    except Exception:
        pass

# Optional GPU acceleration (cuGraph from RAPIDS)
HAS_GPU = False
HAS_CUGRAPH = False
cugraph = None
cudf = None

# Verbose GPU detection
import os
import sys
import platform
VERBOSE_GPU = os.getenv('VERBOSE_GPU', '1') == '1'
is_windows = platform.system() == 'Windows'
is_mac = platform.system() == 'Darwin'

if VERBOSE_GPU:
    if is_windows:
        print("[graph_tasks] Running on Windows - using NetworkX for graph analytics (CPU)")
        print("[graph_tasks] For GPU acceleration, use Docker or WSL2 (see WINDOWS_GPU.md)")
    elif is_mac:
        print("[graph_tasks] Running on macOS - using NetworkX for graph analytics (CPU)")
        print("[graph_tasks] Note: RAPIDS/cuGraph does not support macOS (Intel or Apple Silicon)")
        print("[graph_tasks] For GPU acceleration, use Linux with NVIDIA GPU or cloud instances")
    else:
        print("[graph_tasks] GPU Detection Starting...")

try:
    import cudf  # GPU DataFrame
    import cugraph  # GPU graph analytics
    HAS_CUGRAPH = True
    if VERBOSE_GPU:
        print(f"[graph_tasks] ✓ cuGraph available (version {cugraph.__version__})")
        print(f"[graph_tasks] ✓ cuDF available (version {cudf.__version__})")
except ImportError as e:
    HAS_GPU = False
    HAS_CUGRAPH = False
    if VERBOSE_GPU and not is_windows and not is_mac:
        # Only show detailed cuDF errors on Linux (where it's expected to work)
        print(f"[graph_tasks] ✗ cuGraph/cuDF not installed")

        py_version = sys.version_info
        py_supported = (3, 10) <= py_version[:2] <= (3, 12)

        if not py_supported:
            print(f"[graph_tasks]   ⚠ Python {py_version.major}.{py_version.minor} not supported by RAPIDS")
            print(f"[graph_tasks]   RAPIDS requires Python 3.10, 3.11, or 3.12")
            print("[graph_tasks]   Solution: Use Docker (has Python 3.11)")
        else:
            print("[graph_tasks]   Install: pip install cudf-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com")
            print("[graph_tasks]   Or use: docker-compose -f docker-compose.prod.gpu.yml up")
        print("[graph_tasks]   Falling back to NetworkX (CPU)")

if HAS_CUGRAPH:
    try:
        import torch  # Check CUDA availability
        HAS_GPU = torch.cuda.is_available()
        if HAS_GPU:
            print(f"[graph_tasks] ✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"[graph_tasks] ✓ CUDA version: {torch.version.cuda}")
            print(f"[graph_tasks] ✓ GPU GRAPH ACCELERATION ENABLED")
        else:
            if VERBOSE_GPU:
                print("[graph_tasks] ✗ cuGraph installed but no CUDA GPU detected")
                print("[graph_tasks]   Check: nvidia-smi")
    except ImportError:
        HAS_GPU = False
        if VERBOSE_GPU:
            print("[graph_tasks] ✗ PyTorch not installed (cannot detect GPU)")

if not HAS_GPU and VERBOSE_GPU:
    if is_windows:
        print("[graph_tasks] ✓ NetworkX ready for graph analytics (works great on CPU)")
        print("[graph_tasks]   Set VERBOSE_GPU=0 to hide this message")
    elif is_mac:
        print("[graph_tasks] ✓ NetworkX ready for graph analytics (CPU mode)")
        print("[graph_tasks]   Set VERBOSE_GPU=0 to hide this message")
    else:
        print("[graph_tasks] ⚠ GPU graph acceleration disabled - using NetworkX (CPU)")
        print("[graph_tasks]   Set VERBOSE_GPU=0 to hide these messages")



# ---------------------------- Loading / Encoding ---------------------------- #

def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    # Try common aliases
    src = cols.get("src") or cols.get("source") or cols.get("from") or list(df.columns)[0]
    dst = cols.get("dst") or cols.get("target") or cols.get("to") or list(df.columns)[1]
    df = df.rename(columns={src: "src", dst: "dst"})
    # Optional columns
    if "weight" in cols:
        df = df.rename(columns={cols["weight"]: "weight"})
    if "ts" in cols or "timestamp" in cols or "time" in cols:
        df = df.rename(columns={cols.get("ts", cols.get("timestamp", cols.get("time"))): "ts"})
    if "etype" in cols or "type" in cols:
        df = df.rename(columns={cols.get("etype", cols.get("type")): "etype"})
    return df[["src", "dst"] + [c for c in ["weight", "ts", "etype"] if c in df.columns]]


def load_edges_csv(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if df.shape[1] < 2:
        # Try no-header
        df = pd.read_csv(io.BytesIO(file_bytes), header=None)
    df = _coerce_columns(df)
    df = df.dropna(subset=["src", "dst"]).astype({"src": str, "dst": str})
    return df


def load_edges_json(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    data = json.loads(file_bytes.decode(encoding))
    if isinstance(data, dict) and "edges" in data:
        data = data["edges"]
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of edge objects or have key 'edges'")
    df = pd.DataFrame(data)
    df = _coerce_columns(df)
    df = df.dropna(subset=["src", "dst"]).astype({"src": str, "dst": str})
    return df


def load_nodes_csv(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    """Load node list from CSV with attributes. First column must be node ID."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    if df.shape[1] < 1:
        raise ValueError("Node CSV must have at least one column (id)")

    # Rename first column to 'id' if not already
    cols = df.columns.tolist()
    if cols[0].lower() not in ["id", "node", "node_id"]:
        df = df.rename(columns={cols[0]: "id"})
    else:
        df = df.rename(columns={cols[0]: "id"})

    df["id"] = df["id"].astype(str)
    return df


def load_nodes_json(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    """Load node list from JSON with attributes."""
    data = json.loads(file_bytes.decode(encoding))
    if isinstance(data, dict) and "nodes" in data:
        data = data["nodes"]
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of node objects or have key 'nodes'")
    df = pd.DataFrame(data)

    # Find ID column
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id") or cols.get("node") or cols.get("node_id") or list(df.columns)[0]
    df = df.rename(columns={id_col: "id"})
    df["id"] = df["id"].astype(str)
    return df


def load_edges_edge_file(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load edge list from .edge format (space/tab-separated).
    Format: src dst [weight]
    Lines starting with # are treated as comments.
    """
    lines = file_bytes.decode(encoding).splitlines()
    edges = []

    for line in lines:
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Split by whitespace
        parts = line.split()
        if len(parts) < 2:
            continue

        edge = {"src": parts[0], "dst": parts[1]}
        if len(parts) >= 3:
            try:
                edge["weight"] = float(parts[2])
            except ValueError:
                pass  # Ignore invalid weights

        edges.append(edge)

    if not edges:
        raise ValueError(".edge file contains no valid edges")

    df = pd.DataFrame(edges)
    df["src"] = df["src"].astype(str)
    df["dst"] = df["dst"].astype(str)
    return df


def load_nodes_node_file(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load node list from .node format (space/tab-separated).
    Format: id [label] [attr1] [attr2] ...
    Lines starting with # are treated as comments.
    """
    lines = file_bytes.decode(encoding).splitlines()
    nodes = []
    max_cols = 0

    # First pass: determine structure
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        max_cols = max(max_cols, len(parts))

    # Second pass: parse nodes
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) < 1:
            continue

        node = {"id": parts[0]}

        # If there's a second column, treat it as label
        if len(parts) >= 2:
            node["label"] = parts[1]

        # Additional columns as attr1, attr2, etc.
        for i in range(2, len(parts)):
            node[f"attr{i-1}"] = parts[i]

        nodes.append(node)

    if not nodes:
        raise ValueError(".node file contains no valid nodes")

    df = pd.DataFrame(nodes)
    df["id"] = df["id"].astype(str)
    return df


def load_circles_file(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load social circles from .circles format.
    Format: circleName\tnode1 node2 node3...
    or: circleName node1 node2 node3...
    Returns DataFrame with columns: circle, node
    """
    lines = file_bytes.decode(encoding).splitlines()
    circle_data = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Try tab-separated first, then space-separated
        if '\t' in line:
            parts = line.split('\t', 1)
            if len(parts) == 2:
                circle_name = parts[0]
                nodes = parts[1].split()
        else:
            parts = line.split()
            if len(parts) >= 2:
                circle_name = parts[0]
                nodes = parts[1:]
            else:
                continue

        # Create one row per node in circle
        for node in nodes:
            circle_data.append({"circle": circle_name, "node": node})

    if not circle_data:
        return pd.DataFrame(columns=["circle", "node"])

    df = pd.DataFrame(circle_data)
    df["node"] = df["node"].astype(str)
    return df


def load_feat_file(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load node features from .feat format.
    Format: nodeID feat1 feat2 feat3... (binary 0/1 features)
    Returns DataFrame with columns: id, f0, f1, f2, ...
    """
    lines = file_bytes.decode(encoding).splitlines()
    feat_data = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        node_id = parts[0]
        features = parts[1:]

        # Create feature dict
        feat_dict = {"id": node_id}
        for i, feat in enumerate(features):
            feat_dict[f"f{i}"] = int(feat) if feat.isdigit() else feat

        feat_data.append(feat_dict)

    if not feat_data:
        raise ValueError(".feat file contains no valid features")

    df = pd.DataFrame(feat_data)
    df["id"] = df["id"].astype(str)
    return df


def load_egofeat_file(file_bytes: bytes, encoding: str = "utf-8") -> T.List[int]:
    """
    Load ego node features from .egofeat format.
    Format: feat1 feat2 feat3... (single line, binary 0/1 features)
    Returns list of feature values.
    """
    lines = file_bytes.decode(encoding).splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        return [int(f) if f.isdigit() else f for f in parts]

    raise ValueError(".egofeat file contains no valid features")


def load_featnames_file(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load feature names from .featnames format.
    Format: featureID featureName [anonymized]
    or: featureName anonymized
    Returns DataFrame with columns: feature_id, feature_name, anonymized
    """
    lines = file_bytes.decode(encoding).splitlines()
    featnames_data = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) == 0:
            continue

        # Try to parse: featureID featureName [anonymized]
        if len(parts) >= 3:
            feat_dict = {
                "feature_id": i,
                "feature_name": parts[0] + " " + parts[1],
                "anonymized": parts[2] if len(parts) > 2 else ""
            }
        elif len(parts) == 2:
            feat_dict = {
                "feature_id": i,
                "feature_name": parts[0],
                "anonymized": parts[1]
            }
        else:
            feat_dict = {
                "feature_id": i,
                "feature_name": parts[0],
                "anonymized": ""
            }

        featnames_data.append(feat_dict)

    if not featnames_data:
        return pd.DataFrame(columns=["feature_id", "feature_name", "anonymized"])

    return pd.DataFrame(featnames_data)


@dataclass
class GraphData:
    n: int
    edges: pd.DataFrame              # columns: src_idx, dst_idx, (optional) weight
    id_to_idx: Dict[str, int]
    idx_to_id: List[str]
    out_adj: List[List[int]]         # adjacency lists (outgoing)
    in_adj: List[List[int]]          # adjacency lists (incoming)
    gb_matrix: Optional["gb.Matrix"] # GraphBLAS adjacency (BOOL), if available
    nodes: Optional[pd.DataFrame] = None       # node attributes with 'id' column + custom attributes
    circles: Optional[pd.DataFrame] = None     # social circles with 'circle' and 'node' columns
    features: Optional[pd.DataFrame] = None    # node features with 'id' and f0, f1, f2... columns
    featnames: Optional[pd.DataFrame] = None   # feature names with 'feature_id', 'feature_name', 'anonymized'
    ego_id: Optional[str] = None               # ego node ID if this is an ego network
    ego_features: Optional[T.List] = None      # ego node features if available


def encode_nodes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """
    Map string node ids -> contiguous [0..n) indices.
    """
    # Use pandas.factorize for stable integer coding over src+dst
    all_nodes = pd.Index(df["src"]).append(pd.Index(df["dst"])).astype(str)
    codes, uniques = pd.factorize(all_nodes, sort=False)
    uniq = uniques.tolist()
    id_to_idx = {node: i for i, node in enumerate(uniq)}
    # Map on the original frame
    src_idx = [id_to_idx[s] for s in df["src"].astype(str)]
    dst_idx = [id_to_idx[d] for d in df["dst"].astype(str)]
    out = df.copy()
    out["src_idx"] = src_idx
    out["dst_idx"] = dst_idx
    return out, id_to_idx, uniq


def build_graph(
    df_edges: pd.DataFrame,
    df_nodes: Optional[pd.DataFrame] = None,
    df_circles: Optional[pd.DataFrame] = None,
    df_features: Optional[pd.DataFrame] = None,
    df_featnames: Optional[pd.DataFrame] = None,
    ego_id: Optional[str] = None,
    ego_features: Optional[T.List] = None,
    use_weights: bool = False
) -> GraphData:
    df = df_edges.copy()
    df, id_to_idx, idx_to_id = encode_nodes(df)
    n = len(idx_to_id)

    # Normalize to required columns
    needed = ["src_idx", "dst_idx"]
    if use_weights and "weight" in df.columns:
        needed.append("weight")
    df = df[needed]

    # Build adjacency lists
    out_adj = [[] for _ in range(n)]
    in_adj = [[] for _ in range(n)]
    for s, d in zip(df["src_idx"].to_numpy(), df["dst_idx"].to_numpy()):
        out_adj[s].append(int(d))
        in_adj[d].append(int(s))

    # Optional GraphBLAS matrix
    gb_matrix = None
    if HAS_GB:
        try:
            rows = df["src_idx"].to_numpy(dtype=np.int64)
            cols = df["dst_idx"].to_numpy(dtype=np.int64)
            vals = np.ones(len(df), dtype=np.bool_)

            # Try newer API first (python-graphblas >= 2023.x)
            try:
                # Newer API: Matrix.from_coo (requires scipy)
                try:
                    from scipy.sparse import coo_matrix
                    coo = coo_matrix((vals, (rows, cols)), shape=(n, n))
                    gb_matrix = gb.Matrix.from_coo(coo)
                except ImportError:
                    # scipy not installed, skip GraphBLAS
                    gb_matrix = None
            except (AttributeError, TypeError):
                # Older API: Matrix.from_lists (deprecated in newer versions)
                try:
                    gb_matrix = gb.Matrix.from_lists(rows, cols, vals, n, n, gb.types.BOOL)  # type: ignore
                except (AttributeError, TypeError):
                    # Neither API works - GraphBLAS version incompatible
                    gb_matrix = None
        except Exception as e:
            # GraphBLAS is optional - if it fails, just continue without it
            if VERBOSE_GPU:
                print(f"[graph_tasks] GraphBLAS matrix creation failed: {e}")
            gb_matrix = None

    # Process node attributes if provided
    nodes_df = None
    if df_nodes is not None:
        # Ensure nodes have the index column
        nodes_df = df_nodes.copy()
        nodes_df["idx"] = nodes_df["id"].map(id_to_idx)
        # Keep nodes that exist in the graph
        nodes_df = nodes_df.dropna(subset=["idx"])
        nodes_df["idx"] = nodes_df["idx"].astype(int)
        nodes_df = nodes_df.sort_values("idx").reset_index(drop=True)

    return GraphData(
        n=n,
        edges=df,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        out_adj=out_adj,
        in_adj=in_adj,
        gb_matrix=gb_matrix,
        nodes=nodes_df,
        circles=df_circles,
        features=df_features,
        featnames=df_featnames,
        ego_id=ego_id,
        ego_features=ego_features,
    )


# ------------------------------- GPU Helpers ------------------------------- #

def _to_cugraph(g: GraphData):
    """Convert GraphData to cuGraph format (GPU DataFrame)."""
    if not HAS_CUGRAPH:
        raise RuntimeError("cuGraph not available")

    # Create cuDF DataFrame from edges
    edge_df = cudf.DataFrame({
        'src': g.edges['src_idx'].values,
        'dst': g.edges['dst_idx'].values
    })

    # Create cuGraph object
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(edge_df, source='src', destination='dst')

    return G


# ------------------------- GPU-Accelerated Algorithms ---------------------- #

def degrees_gpu(g: GraphData) -> pd.DataFrame:
    """GPU-accelerated degree computation using cuGraph."""
    try:
        G = _to_cugraph(g)

        # Compute in-degree and out-degree using cuGraph
        in_deg = G.in_degree()
        out_deg = G.out_degree()

        # Convert to pandas and merge
        in_deg_pd = in_deg.to_pandas().rename(columns={'degree': 'in_degree'})
        out_deg_pd = out_deg.to_pandas().rename(columns={'degree': 'out_degree'})

        # Merge on vertex
        result = in_deg_pd.merge(out_deg_pd, on='vertex', how='outer').fillna(0)
        result['degree'] = result['in_degree'] + result['out_degree']

        # Map vertex indices to node IDs
        result['node'] = result['vertex'].apply(lambda x: g.idx_to_id[int(x)] if int(x) < len(g.idx_to_id) else str(x))

        # Select and order columns
        result = result[['node', 'out_degree', 'in_degree', 'degree']]
        result['out_degree'] = result['out_degree'].astype(int)
        result['in_degree'] = result['in_degree'].astype(int)
        result['degree'] = result['degree'].astype(int)
        result.sort_values('degree', ascending=False, inplace=True, ignore_index=True)

        return result
    except Exception as e:
        print(f"[graph_tasks] GPU degrees failed: {e}, falling back to CPU")
        return degrees_cpu(g)


def pagerank_gpu(
    g: GraphData,
    alpha: float = 0.85,
    iters: int = 40,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """GPU-accelerated PageRank using cuGraph."""
    try:
        G = _to_cugraph(g)

        # Compute PageRank on GPU
        pr_df = cugraph.pagerank(G, alpha=alpha, max_iter=iters, tol=tol)

        # Convert to pandas
        pr_pd = pr_df.to_pandas()

        # Map vertex indices to node IDs
        pr_pd['node'] = pr_pd['vertex'].apply(lambda x: g.idx_to_id[int(x)] if int(x) < len(g.idx_to_id) else str(x))

        # Select and order columns
        result = pr_pd[['node', 'pagerank']].rename(columns={'pagerank': 'pr'})
        result.sort_values('pr', ascending=False, inplace=True, ignore_index=True)

        return result
    except Exception as e:
        print(f"[graph_tasks] GPU pagerank failed: {e}, falling back to CPU")
        return pagerank_cpu(g, alpha, iters, tol)


def bfs_gpu(g: GraphData, source_node: str) -> pd.DataFrame:
    """GPU-accelerated BFS using cuGraph."""
    try:
        if source_node not in g.id_to_idx:
            raise KeyError(f"Unknown source node: {source_node}")

        G = _to_cugraph(g)
        source_idx = g.id_to_idx[source_node]

        # Compute BFS on GPU
        bfs_df = cugraph.bfs(G, start=source_idx)

        # Convert to pandas
        bfs_pd = bfs_df.to_pandas()

        # Map vertex indices to node IDs
        bfs_pd['node'] = bfs_pd['vertex'].apply(lambda x: g.idx_to_id[int(x)] if int(x) < len(g.idx_to_id) else str(x))

        # Select columns and handle unreachable nodes
        result = bfs_pd[['node', 'distance']]
        result['distance'] = result['distance'].fillna(-1).astype(int)

        return result
    except Exception as e:
        print(f"[graph_tasks] GPU bfs failed: {e}, falling back to CPU")
        return bfs_cpu(g, source_node)


def triangles_gpu(g: GraphData, max_nodes: int = 20000) -> Dict[str, int]:
    """GPU-accelerated triangle counting using cuGraph."""
    try:
        if g.n > max_nodes:
            return {"triangles": -1, "note": f"skipped (n={g.n} > max_nodes={max_nodes})"}

        # cuGraph triangle counting expects undirected graph
        edge_df = cudf.DataFrame({
            'src': g.edges['src_idx'].values,
            'dst': g.edges['dst_idx'].values
        })

        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(edge_df, source='src', destination='dst')

        # Count triangles on GPU
        triangle_count = cugraph.triangle_count(G)

        # Sum up all triangles (each triangle counted 3 times, once per vertex)
        total_triangles = int(triangle_count.to_pandas()['counts'].sum() // 3)

        return {"triangles": total_triangles}
    except Exception as e:
        print(f"[graph_tasks] GPU triangles failed: {e}, falling back to CPU")
        return triangles_cpu(g, max_nodes)


# ---------------------------- CPU Algorithms (Fallback) -------------------- #
# Uses NetworkX for reliable, cross-platform graph algorithms

def _graphdata_to_networkx(g: GraphData) -> nx.DiGraph:
    """Convert GraphData to NetworkX DiGraph for algorithm processing."""
    G = nx.DiGraph()
    # Add nodes with original IDs
    G.add_nodes_from(g.idx_to_id)
    # Add edges using original node IDs
    edges = [(g.idx_to_id[row.src_idx], g.idx_to_id[row.dst_idx])
             for row in g.edges.itertuples()]
    G.add_edges_from(edges)
    return G

def degrees_cpu(g: GraphData) -> pd.DataFrame:
    """CPU implementation of degrees (original)."""
    outdeg = np.fromiter((len(g.out_adj[i]) for i in range(g.n)), dtype=np.int64, count=g.n)
    indeg = np.fromiter((len(g.in_adj[i]) for i in range(g.n)), dtype=np.int64, count=g.n)
    res = pd.DataFrame({
        "node": g.idx_to_id,
        "out_degree": outdeg,
        "in_degree": indeg,
        "degree": outdeg + indeg,
    })
    res.sort_values("degree", ascending=False, inplace=True, ignore_index=True)
    return res


def degrees(g: GraphData) -> pd.DataFrame:
    """Compute node degrees. Automatically uses GPU if available."""
    if HAS_GPU and HAS_CUGRAPH:
        return degrees_gpu(g)
    else:
        return degrees_cpu(g)


def pagerank_cpu(
    g: GraphData,
    alpha: float = 0.85,
    iters: int = 40,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    CPU implementation of PageRank using NetworkX (cross-platform, reliable).
    Returns DataFrame with columns: node, pr.
    """
    if g.n == 0:
        return pd.DataFrame(columns=["node", "pr"])

    # Convert to NetworkX graph
    G = _graphdata_to_networkx(g)

    # Compute PageRank using NetworkX
    pr_dict = nx.pagerank(G, alpha=alpha, max_iter=iters, tol=tol)

    # Convert to DataFrame and sort
    df = pd.DataFrame(list(pr_dict.items()), columns=["node", "pr"])
    return df.sort_values("pr", ascending=False, ignore_index=True)


def pagerank(
    g: GraphData,
    alpha: float = 0.85,
    iters: int = 40,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Compute PageRank. Automatically uses GPU if available."""
    if HAS_GPU and HAS_CUGRAPH:
        return pagerank_gpu(g, alpha, iters, tol)
    else:
        return pagerank_cpu(g, alpha, iters, tol)


def bfs_cpu(g: GraphData, source_node: str) -> pd.DataFrame:
    """
    CPU implementation of breadth-first search distances from a source node (by original id).
    For directed graphs, traversal follows outgoing edges.
    """
    if source_node not in g.id_to_idx:
        raise KeyError(f"Unknown source node: {source_node}")
    s = g.id_to_idx[source_node]
    dist = np.full(g.n, -1, dtype=np.int64)
    q: List[int] = [s]
    dist[s] = 0
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        du = dist[u] + 1
        for v in g.out_adj[u]:
            if dist[v] == -1:
                dist[v] = du
                q.append(v)
    return pd.DataFrame({"node": g.idx_to_id, "distance": dist})


def bfs(g: GraphData, source_node: str) -> pd.DataFrame:
    """Compute BFS distances from source node. Automatically uses GPU if available."""
    if HAS_GPU and HAS_CUGRAPH:
        return bfs_gpu(g, source_node)
    else:
        return bfs_cpu(g, source_node)


def triangles_cpu(g: GraphData, max_nodes: int = 20000) -> Dict[str, int]:
    """
    CPU implementation of triangle counting on an undirected simple graph via node-ordered set intersections.
    Skips if n > max_nodes (to avoid O(m * sqrt(m)) blowups).
    """
    n = g.n
    if n == 0:
        return {"triangles": 0}
    if n > max_nodes:
        return {"triangles": -1, "note": f"skipped (n={n} > max_nodes={max_nodes})"}

    # Build symmetric adjacency sets
    nbrs: List[set[int]] = [set() for _ in range(n)]
    for u in range(n):
        for v in g.out_adj[u]:
            nbrs[u].add(v)
            nbrs[v].add(u)

    # Node ordering by degree (forward algorithm)
    order = sorted(range(n), key=lambda i: len(nbrs[i]))
    rank = {u: i for i, u in enumerate(order)}
    forward = [set() for _ in range(n)]
    for u in range(n):
        forward[u] = {v for v in nbrs[u] if rank[u] < rank[v]}

    # Count intersections
    tri = 0
    for u in range(n):
        fu = forward[u]
        if not fu:
            continue
        for v in fu:
            tri += len(fu & forward[v])

    return {"triangles": tri}


def triangles_undirected(g: GraphData, max_nodes: int = 20000) -> Dict[str, int]:
    """Count triangles in undirected graph. Automatically uses GPU if available."""
    if HAS_GPU and HAS_CUGRAPH:
        return triangles_gpu(g, max_nodes)
    else:
        return triangles_cpu(g, max_nodes)


# ---------------------------- Eigenvector Centrality ----------------------- #

def eigenvector_gpu(g: GraphData, max_iter: int = 100, tol: float = 1e-6) -> pd.DataFrame:
    """GPU-accelerated eigenvector centrality using cuGraph."""
    try:
        G = _to_cugraph(g)

        # Compute eigenvector centrality on GPU
        ec_df = cugraph.eigenvector_centrality(G, max_iter=max_iter, tol=tol)

        # Convert to pandas
        ec_pd = ec_df.to_pandas()

        # Map vertex indices to node IDs
        ec_pd['node'] = ec_pd['vertex'].apply(lambda x: g.idx_to_id[int(x)] if int(x) < len(g.idx_to_id) else str(x))

        # Select and order columns
        result = ec_pd[['node', 'eigenvector_centrality']].rename(columns={'eigenvector_centrality': 'centrality'})
        result.sort_values('centrality', ascending=False, inplace=True, ignore_index=True)

        return result
    except Exception as e:
        print(f"[graph_tasks] GPU eigenvector failed: {e}, falling back to CPU")
        return eigenvector_cpu(g, max_iter, tol)


def eigenvector_cpu(g: GraphData, max_iter: int = 100, tol: float = 1e-6) -> pd.DataFrame:
    """
    CPU implementation of eigenvector centrality using NetworkX (cross-platform, reliable).
    Returns DataFrame with columns: node, centrality.
    """
    if g.n == 0:
        return pd.DataFrame(columns=["node", "centrality"])

    # Convert to NetworkX graph
    G = _graphdata_to_networkx(g)

    try:
        # Compute eigenvector centrality using NetworkX
        ec_dict = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        # If power iteration fails (disconnected graph, etc.), return zeros
        ec_dict = {node: 0.0 for node in G.nodes()}

    # Convert to DataFrame and sort
    df = pd.DataFrame(list(ec_dict.items()), columns=["node", "centrality"])
    return df.sort_values("centrality", ascending=False, ignore_index=True)


def eigenvector_centrality(g: GraphData, max_iter: int = 100, tol: float = 1e-6) -> pd.DataFrame:
    """Compute eigenvector centrality. Automatically uses GPU if available."""
    if HAS_GPU and HAS_CUGRAPH:
        return eigenvector_gpu(g, max_iter, tol)
    else:
        return eigenvector_cpu(g, max_iter, tol)


# ---------------------------- Betweenness Centrality ----------------------- #

def betweenness_gpu(g: GraphData, normalized: bool = True) -> pd.DataFrame:
    """GPU-accelerated betweenness centrality using cuGraph."""
    try:
        G = _to_cugraph(g)

        # Compute betweenness centrality on GPU
        bc_df = cugraph.betweenness_centrality(G, normalized=normalized)

        # Convert to pandas
        bc_pd = bc_df.to_pandas()

        # Map vertex indices to node IDs
        bc_pd['node'] = bc_pd['vertex'].apply(lambda x: g.idx_to_id[int(x)] if int(x) < len(g.idx_to_id) else str(x))

        # Select and order columns
        result = bc_pd[['node', 'betweenness_centrality']].rename(columns={'betweenness_centrality': 'centrality'})
        result.sort_values('centrality', ascending=False, inplace=True, ignore_index=True)

        return result
    except Exception as e:
        print(f"[graph_tasks] GPU betweenness failed: {e}, falling back to CPU")
        return betweenness_cpu(g, normalized)


def betweenness_cpu(g: GraphData, normalized: bool = True) -> pd.DataFrame:
    """
    CPU implementation of betweenness centrality using NetworkX (cross-platform, reliable).
    Returns DataFrame with columns: node, centrality.
    """
    if g.n == 0:
        return pd.DataFrame(columns=["node", "centrality"])

    # Convert to NetworkX graph
    G = _graphdata_to_networkx(g)

    # Compute betweenness centrality using NetworkX (uses Brandes' algorithm)
    bc_dict = nx.betweenness_centrality(G, normalized=normalized)

    # Convert to DataFrame and sort
    df = pd.DataFrame(list(bc_dict.items()), columns=["node", "centrality"])
    return df.sort_values("centrality", ascending=False, ignore_index=True)


def betweenness_centrality(g: GraphData, normalized: bool = True) -> pd.DataFrame:
    """Compute betweenness centrality. Automatically uses GPU if available."""
    if HAS_GPU and HAS_CUGRAPH:
        return betweenness_gpu(g, normalized)
    else:
        return betweenness_cpu(g, normalized)


# ------------------------------ Runner Helpers ----------------------------- #

def merge_node_attributes(result_df: pd.DataFrame, g: GraphData) -> pd.DataFrame:
    """
    Merge node attributes from graph into result DataFrame.
    Assumes result_df has a 'node' column with node IDs.
    """
    if g.nodes is None:
        return result_df
    # Join on node ID
    merged = result_df.merge(g.nodes, left_on="node", right_on="id", how="left")
    # Drop duplicate id column if present
    if "id" in merged.columns and "node" in merged.columns:
        merged = merged.drop(columns=["id"])
    # Drop idx column if present
    if "idx" in merged.columns:
        merged = merged.drop(columns=["idx"])
    return merged


def load_graph_from_bytes(
    file_bytes: bytes,
    kind: str,
    nodes_bytes: Optional[bytes] = None,
    nodes_kind: Optional[str] = None
) -> GraphData:
    """
    Load graph from edge list with optional node attributes.
    kind, nodes_kind ∈ {"csv","json","edge","node"}
    """
    if kind == "csv":
        df_edges = load_edges_csv(file_bytes)
    elif kind == "json":
        df_edges = load_edges_json(file_bytes)
    elif kind == "edge":
        df_edges = load_edges_edge_file(file_bytes)
    else:
        raise ValueError("kind must be 'csv', 'json', or 'edge'")

    df_nodes = None
    if nodes_bytes:
        nk = nodes_kind or kind  # Default to same format as edges
        if nk == "csv":
            df_nodes = load_nodes_csv(nodes_bytes)
        elif nk == "json":
            df_nodes = load_nodes_json(nodes_bytes)
        elif nk == "node":
            df_nodes = load_nodes_node_file(nodes_bytes)
        else:
            raise ValueError("nodes_kind must be 'csv', 'json', or 'node'")

    return build_graph(df_edges, df_nodes=df_nodes)


def load_ego_network_from_files(
    files_dict: Dict[str, bytes],
    ego_id: Optional[str] = None
) -> GraphData:
    """
    Load a complete ego network from multiple files.

    Expected files_dict keys (all optional except edges):
    - 'edges' or '.edges': Edge list
    - 'circles' or '.circles': Social circles
    - 'feat' or '.feat': Node features
    - 'egofeat' or '.egofeat': Ego node features
    - 'featnames' or '.featnames': Feature names
    - 'node' or '.node': Node attributes

    Args:
        files_dict: Dictionary mapping file type to bytes content
        ego_id: Optional ego node ID (can be inferred from filename)

    Returns:
        GraphData with all available components
    """
    # Parse edges (required)
    edges_bytes = files_dict.get('edges') or files_dict.get('.edges')
    if not edges_bytes:
        raise ValueError("Edges file is required")

    df_edges = load_edges_edge_file(edges_bytes)

    # Parse optional components
    df_circles = None
    circles_bytes = files_dict.get('circles') or files_dict.get('.circles')
    if circles_bytes:
        try:
            df_circles = load_circles_file(circles_bytes)
        except Exception:
            pass

    df_features = None
    feat_bytes = files_dict.get('feat') or files_dict.get('.feat')
    if feat_bytes:
        try:
            df_features = load_feat_file(feat_bytes)
        except Exception:
            pass

    ego_features = None
    egofeat_bytes = files_dict.get('egofeat') or files_dict.get('.egofeat')
    if egofeat_bytes:
        try:
            ego_features = load_egofeat_file(egofeat_bytes)
        except Exception:
            pass

    df_featnames = None
    featnames_bytes = files_dict.get('featnames') or files_dict.get('.featnames')
    if featnames_bytes:
        try:
            df_featnames = load_featnames_file(featnames_bytes)
        except Exception:
            pass

    df_nodes = None
    node_bytes = files_dict.get('node') or files_dict.get('.node')
    if node_bytes:
        try:
            df_nodes = load_nodes_node_file(node_bytes)
        except Exception:
            pass

    # Merge features with featnames if both available
    if df_features is not None and df_featnames is not None:
        # Rename feature columns based on featnames
        for i, row in df_featnames.iterrows():
            old_col = f"f{i}"
            if old_col in df_features.columns:
                new_name = row['feature_name'] if row['feature_name'] else old_col
                df_features = df_features.rename(columns={old_col: new_name})

    # Merge features into nodes if nodes exist
    if df_nodes is not None and df_features is not None:
        df_nodes = df_nodes.merge(df_features, on="id", how="left")
    elif df_features is not None:
        df_nodes = df_features

    return build_graph(
        df_edges,
        df_nodes=df_nodes,
        df_circles=df_circles,
        df_features=df_features,
        df_featnames=df_featnames,
        ego_id=ego_id,
        ego_features=ego_features
    )


def run_graph_metrics(
    g: GraphData,
    tasks: Iterable[str],
    *,
    pagerank_alpha: float = 0.85,
    pagerank_iters: int = 40,
    pagerank_tol: float = 1e-6,
    bfs_source: Optional[str] = None,
    triangles_limit: int = 20000,
) -> Dict[str, object]:
    """
    tasks: subset of {"degrees","pagerank","bfs","triangles"}
    """
    out: Dict[str, object] = {"n_nodes": g.n, "n_edges": int(g.edges.shape[0])}
    for t in tasks:
        t = t.lower().strip()
        if t == "degrees":
            out["degrees"] = degrees(g).to_dict(orient="records")
        elif t == "pagerank":
            pr = pagerank(g, alpha=pagerank_alpha, iters=pagerank_iters, tol=pagerank_tol)
            out["pagerank"] = pr.to_dict(orient="records")
        elif t == "bfs":
            if not bfs_source:
                raise ValueError("bfs_source must be provided for BFS")
            out["bfs"] = bfs(g, bfs_source).to_dict(orient="records")
        elif t == "triangles":
            out["triangles"] = triangles_undirected(g, max_nodes=triangles_limit)
        else:
            raise ValueError(f"Unknown graph task: {t}")
    return out
