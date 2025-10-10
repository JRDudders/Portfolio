"""
graph_tasks.py — lightweight graph analytics with optional GraphBLAS

What it does
------------
- Loads edge lists from CSV/JSON bytes
- Builds compact integer node index maps
- Computes degrees, PageRank, BFS, and (optionally) triangle counts
- Uses efficient pure-Python/NumPy routines by default
- If pygraphblas is installed, exposes the adjacency as a GraphBLAS matrix
  (you can extend algorithms to use it later)

Edge CSV schema (auto-detected):
  src, dst[, weight, ts, etype]

Edge JSON schema:
  Either a list of {"src": "...", "dst": "...", ...} objects
  or a dict with key "edges" -> list of such objects.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

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


@dataclass
class GraphData:
    n: int
    edges: pd.DataFrame              # columns: src_idx, dst_idx, (optional) weight
    id_to_idx: Dict[str, int]
    idx_to_id: List[str]
    out_adj: List[List[int]]         # adjacency lists (outgoing)
    in_adj: List[List[int]]          # adjacency lists (incoming)
    gb_matrix: Optional["gb.Matrix"] # GraphBLAS adjacency (BOOL), if available


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


def build_graph(df_edges: pd.DataFrame, use_weights: bool = False) -> GraphData:
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
        rows = df["src_idx"].to_numpy(dtype=np.int64)
        cols = df["dst_idx"].to_numpy(dtype=np.int64)
        vals = np.ones(len(df), dtype=np.bool_)
        # from_lists: (rows, cols, vals, nrows, ncols, typ)
        gb_matrix = gb.Matrix.from_lists(rows, cols, vals, n, n, gb.types.BOOL)  # type: ignore

    return GraphData(
        n=n,
        edges=df,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        out_adj=out_adj,
        in_adj=in_adj,
        gb_matrix=gb_matrix,
    )


# ------------------------------- Algorithms -------------------------------- #

def degrees(g: GraphData) -> pd.DataFrame:
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


def pagerank(
    g: GraphData,
    alpha: float = 0.85,
    iters: int = 40,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Power iteration PageRank on adjacency lists (robust without SciPy).
    Returns DataFrame with columns: node, pr.
    """
    n = g.n
    if n == 0:
        return pd.DataFrame(columns=["node", "pr"])

    outdeg = np.fromiter((len(g.out_adj[i]) for i in range(n)), dtype=np.float64, count=n)
    r = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - alpha) / n

    # Precompute in-neighbor lists for faster iteration
    in_adj = g.in_adj

    for _ in range(iters):
        r_new = np.full(n, teleport, dtype=np.float64)
        for i in range(n):
            # Distribute rank of in-neighbors
            val = 0.0
            for j in in_adj[i]:
                if outdeg[j] > 0.0:
                    val += r[j] / outdeg[j]
            r_new[i] += alpha * val
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new

    # Normalize
    s = r.sum()
    if s > 0:
        r = r / s

    return pd.DataFrame({"node": g.idx_to_id, "pr": r}).sort_values("pr", ascending=False, ignore_index=True)


def bfs(g: GraphData, source_node: str) -> pd.DataFrame:
    """
    Breadth-first search distances from a source node (by original id).
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


def triangles_undirected(g: GraphData, max_nodes: int = 20000) -> Dict[str, int]:
    """
    Counts global triangles on an undirected simple graph via node-ordered set intersections.
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


# ------------------------------ Runner Helpers ----------------------------- #

def load_graph_from_bytes(file_bytes: bytes, kind: str) -> GraphData:
    """
    kind ∈ {"csv","json"}
    """
    if kind == "csv":
        df = load_edges_csv(file_bytes)
    elif kind == "json":
        df = load_edges_json(file_bytes)
    else:
        raise ValueError("kind must be 'csv' or 'json'")
    return build_graph(df)


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
