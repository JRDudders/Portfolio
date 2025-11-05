"""
Graph Analytics Microservice

Handles all graph-related tasks:
- Graph loading (edge lists, ego networks)
- Graph metrics (PageRank, BFS, degrees, triangles)
- Centrality measures (betweenness, eigenvector)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import uvicorn
import os

# Import URL fetch utility
from url_fetch import fetch_url, guess_file_extension

# Import graph processing modules
from graph_processor import (
    load_graph,
    load_ego_network,
    compute_metrics,
    compute_degrees,
    compute_pagerank,
    compute_bfs,
    compute_triangles,
    compute_eigenvector_centrality,
    compute_betweenness_centrality,
    get_graph_info
)

# Import social media edge extraction
from graph_tasks import extract_social_media_edges
import pandas as pd
import io

app = FastAPI(
    title="CiceroWatch Graph Analytics Service",
    description="Graph analytics microservice with GPU acceleration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helper Functions
# ============================================================

def _detect_file_kind(filename: Optional[str]) -> str:
    """Detect file type from extension"""
    if not filename:
        return "csv"

    filename = filename.lower()
    if filename.endswith('.json'):
        return "json"
    elif filename.endswith('.edge') or filename.endswith('.edges'):
        return "edge"
    elif filename.endswith('.node') or filename.endswith('.nodes'):
        return "node"
    else:
        return "csv"


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "service": "graph",
        "status": "healthy",
        "version": "1.0.0"
    }


# ============================================================
# Graph Loading Endpoints
# ============================================================

@app.post("/load")
async def load_graph_file(
    edges_file: UploadFile = File(..., description="Edge list (CSV/JSON/.edge)"),
    nodes_file: Optional[UploadFile] = File(None, description="Optional node attributes"),
):
    """
    Load and validate a graph file
    """
    try:
        # Read edges file
        edges_bytes = await edges_file.read()
        edges_kind = _detect_file_kind(edges_file.filename)

        # Read optional nodes file
        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_file_kind(nodes_file.filename)

        # Load graph
        graph = load_graph(edges_bytes, edges_kind, nodes_bytes, nodes_kind)

        # Get graph info
        info = get_graph_info(graph)

        return {
            "success": True,
            "graph": info
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Graph loading failed: {str(e)}")


@app.post("/ego-network")
async def load_ego_network_files(
    files: List[UploadFile] = File(..., description="Ego network files (.edges, .circles, .feat, etc.)"),
    ego_id: Optional[str] = Query(None, description="Ego node ID")
):
    """
    Load ego network from multiple files and compute centrality measures
    """
    try:
        # Read all files
        files_dict = {}
        for file in files:
            content = await file.read()

            # Determine file type from extension
            filename = file.filename.lower()
            if '.edges' in filename:
                files_dict['edges'] = content
            elif '.circles' in filename:
                files_dict['circles'] = content
            elif '.feat' in filename and 'egofeat' not in filename:
                files_dict['feat'] = content
            elif '.egofeat' in filename:
                files_dict['egofeat'] = content
            elif '.featnames' in filename:
                files_dict['featnames'] = content
            elif '.node' in filename:
                files_dict['node'] = content

        if 'edges' not in files_dict:
            raise ValueError("Edges file (.edges) is required")

        # Load ego network
        graph = load_ego_network(files_dict, ego_id=ego_id)

        # Get graph info with centrality measures for visualization
        info = get_graph_info(graph, include_centrality=True)

        return {
            "success": True,
            "ego_network": info
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ego network loading failed: {str(e)}")


# ============================================================
# URL Fetch Endpoints
# ============================================================

class GraphURLRequest(BaseModel):
    """Request model for loading graph from URLs"""
    edges_url: HttpUrl
    nodes_url: Optional[HttpUrl] = None


class EgoNetworkURLRequest(BaseModel):
    """Request model for loading ego network from URLs"""
    edges_url: HttpUrl
    circles_url: Optional[HttpUrl] = None
    feat_url: Optional[HttpUrl] = None
    egofeat_url: Optional[HttpUrl] = None
    featnames_url: Optional[HttpUrl] = None
    ego_id: Optional[str] = None


@app.post("/load-from-url")
async def load_graph_from_url(request: GraphURLRequest):
    """
    Load graph from URLs instead of file uploads
    """
    try:
        # Fetch edges file
        edges_bytes, content_type = await fetch_url(str(request.edges_url))
        ext = guess_file_extension(str(request.edges_url), content_type)
        edges_kind = _detect_file_kind(f"edges{ext}")

        # Fetch nodes file if provided
        nodes_bytes = None
        nodes_kind = None
        if request.nodes_url:
            nodes_bytes, content_type = await fetch_url(str(request.nodes_url))
            ext = guess_file_extension(str(request.nodes_url), content_type)
            nodes_kind = _detect_file_kind(f"nodes{ext}")

        # Load graph
        graph = load_graph(edges_bytes, edges_kind, nodes_bytes, nodes_kind)

        # Get graph info
        info = get_graph_info(graph)

        return {
            "success": True,
            "graph": info
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Graph loading from URL failed: {str(e)}")


@app.post("/ego-network-from-url")
async def load_ego_network_from_url(request: EgoNetworkURLRequest):
    """
    Load ego network from URLs instead of file uploads
    """
    try:
        files_dict = {}

        # Fetch edges (required)
        edges_bytes, _ = await fetch_url(str(request.edges_url))
        files_dict['edges'] = edges_bytes

        # Fetch optional files
        if request.circles_url:
            circles_bytes, _ = await fetch_url(str(request.circles_url))
            files_dict['circles'] = circles_bytes

        if request.feat_url:
            feat_bytes, _ = await fetch_url(str(request.feat_url))
            files_dict['feat'] = feat_bytes

        if request.egofeat_url:
            egofeat_bytes, _ = await fetch_url(str(request.egofeat_url))
            files_dict['egofeat'] = egofeat_bytes

        if request.featnames_url:
            featnames_bytes, _ = await fetch_url(str(request.featnames_url))
            files_dict['featnames'] = featnames_bytes

        # Load ego network
        graph = load_ego_network(files_dict, ego_id=request.ego_id)

        # Get graph info with centrality measures
        info = get_graph_info(graph, include_centrality=True)

        return {
            "success": True,
            "ego_network": info
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ego network loading from URL failed: {str(e)}")


# ============================================================
# Graph Metrics Endpoints
# ============================================================

@app.post("/metrics")
async def calculate_metrics(
    edges_file: UploadFile = File(...),
    nodes_file: Optional[UploadFile] = File(None),
    tasks: str = Query("degrees,pagerank", description="Comma-separated tasks"),
    pagerank_alpha: float = Query(0.85),
    pagerank_iters: int = Query(40),
    pagerank_tol: float = Query(1e-6),
    bfs_source: Optional[str] = Query(None),
    triangles_limit: int = Query(20000)
):
    """
    Compute multiple graph metrics in one call
    """
    try:
        # Load graph
        edges_bytes = await edges_file.read()
        edges_kind = _detect_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_file_kind(nodes_file.filename)

        graph = load_graph(edges_bytes, edges_kind, nodes_bytes, nodes_kind)

        # Parse tasks
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]

        # Compute metrics
        result = compute_metrics(
            graph,
            tasks=task_list,
            pagerank_alpha=pagerank_alpha,
            pagerank_iters=pagerank_iters,
            pagerank_tol=pagerank_tol,
            bfs_source=bfs_source,
            triangles_limit=triangles_limit
        )

        return {
            "success": True,
            "metrics": result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Metrics computation failed: {str(e)}")


@app.post("/degrees")
async def calculate_degrees(
    edges_file: UploadFile = File(...),
    nodes_file: Optional[UploadFile] = File(None),
    include_attrs: bool = Query(True, description="Include node attributes")
):
    """
    Compute node degrees
    """
    try:
        # Load graph
        edges_bytes = await edges_file.read()
        edges_kind = _detect_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_file_kind(nodes_file.filename)

        graph = load_graph(edges_bytes, edges_kind, nodes_bytes, nodes_kind)

        # Compute degrees
        result = compute_degrees(graph, include_node_attrs=include_attrs)

        return {
            "success": True,
            "degrees": result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Degrees computation failed: {str(e)}")


@app.post("/pagerank")
async def calculate_pagerank(
    edges_file: UploadFile = File(...),
    nodes_file: Optional[UploadFile] = File(None),
    alpha: float = Query(0.85, description="Damping factor"),
    iters: int = Query(40, description="Max iterations"),
    tol: float = Query(1e-6, description="Convergence tolerance"),
    include_attrs: bool = Query(True, description="Include node attributes")
):
    """
    Compute PageRank scores
    """
    try:
        # Load graph
        edges_bytes = await edges_file.read()
        edges_kind = _detect_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_file_kind(nodes_file.filename)

        graph = load_graph(edges_bytes, edges_kind, nodes_bytes, nodes_kind)

        # Compute PageRank
        result = compute_pagerank(graph, alpha=alpha, iters=iters, tol=tol, include_node_attrs=include_attrs)

        return {
            "success": True,
            "pagerank": result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PageRank computation failed: {str(e)}")


@app.post("/bfs")
async def calculate_bfs(
    edges_file: UploadFile = File(...),
    nodes_file: Optional[UploadFile] = File(None),
    source: str = Query(..., description="Source node ID"),
    include_attrs: bool = Query(True, description="Include node attributes")
):
    """
    Compute BFS distances from source node
    """
    try:
        # Load graph
        edges_bytes = await edges_file.read()
        edges_kind = _detect_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_file_kind(nodes_file.filename)

        graph = load_graph(edges_bytes, edges_kind, nodes_bytes, nodes_kind)

        # Compute BFS
        result = compute_bfs(graph, source, include_node_attrs=include_attrs)

        return {
            "success": True,
            "bfs": result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"BFS computation failed: {str(e)}")


@app.post("/triangles")
async def calculate_triangles(
    edges_file: UploadFile = File(...),
    nodes_file: Optional[UploadFile] = File(None),
    max_nodes: int = Query(20000, description="Skip if more nodes than this")
):
    """
    Count triangles in undirected graph
    """
    try:
        # Load graph
        edges_bytes = await edges_file.read()
        edges_kind = _detect_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_file_kind(nodes_file.filename)

        graph = load_graph(edges_bytes, edges_kind, nodes_bytes, nodes_kind)

        # Count triangles
        result = compute_triangles(graph, max_nodes=max_nodes)

        return {
            "success": True,
            "triangles": result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Triangle counting failed: {str(e)}")


@app.post("/eigenvector-centrality")
async def calculate_eigenvector(
    edges_file: UploadFile = File(...),
    nodes_file: Optional[UploadFile] = File(None),
    max_iter: int = Query(100, description="Max iterations"),
    tol: float = Query(1e-6, description="Convergence tolerance"),
    include_attrs: bool = Query(True, description="Include node attributes")
):
    """
    Compute eigenvector centrality
    """
    try:
        # Load graph
        edges_bytes = await edges_file.read()
        edges_kind = _detect_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_file_kind(nodes_file.filename)

        graph = load_graph(edges_bytes, edges_kind, nodes_bytes, nodes_kind)

        # Compute eigenvector centrality
        result = compute_eigenvector_centrality(graph, max_iter=max_iter, tol=tol, include_node_attrs=include_attrs)

        return {
            "success": True,
            "eigenvector_centrality": result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Eigenvector centrality computation failed: {str(e)}")


@app.post("/betweenness-centrality")
async def calculate_betweenness(
    edges_file: UploadFile = File(...),
    nodes_file: Optional[UploadFile] = File(None),
    normalized: bool = Query(True, description="Normalize scores"),
    include_attrs: bool = Query(True, description="Include node attributes")
):
    """
    Compute betweenness centrality
    """
    try:
        # Load graph
        edges_bytes = await edges_file.read()
        edges_kind = _detect_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_file_kind(nodes_file.filename)

        graph = load_graph(edges_bytes, edges_kind, nodes_bytes, nodes_kind)

        # Compute betweenness centrality
        result = compute_betweenness_centrality(graph, normalized=normalized, include_node_attrs=include_attrs)

        return {
            "success": True,
            "betweenness_centrality": result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Betweenness centrality computation failed: {str(e)}")


@app.post("/prepare")
async def prepare_social_media_graph(
    file: UploadFile = File(..., description="CSV or XLSX file with social media data"),
    sheet: str = Query("0", description="Sheet name or index for XLSX files"),
    extract_hashtags: bool = Query(False, description="Extract hashtag edges (user->hashtag)")
):
    """
    Convert social media CSV/XLSX into network edge lists.

    Automatically detects columns (author, mentions, replies, retweets, quotes, URLs, hashtags)
    and extracts various types of network relationships:

    - **mention_edges**: User @mentions another user
    - **reply_edges**: User replies to another user
    - **retweet_edges**: User retweets another user
    - **quote_edges**: User quotes another user
    - **domain_edges**: User shares a URL domain
    - **hashtag_edges**: User uses a hashtag (optional)
    - **nodes**: All unique nodes with type annotations (user/domain/hashtag)

    **Column Detection** (case-insensitive, substring matching):
    - Author: author, username, user, screen_name, from, account
    - Text: text, full_text, tweet, content, body
    - Mentions: mentions, mentioned_users, entities_user_mentions
    - Reply: in_reply_to_screen_name, in_reply_to_user, reply_to
    - Retweet: retweeted_user, retweeted_username, rt_username
    - Quote: quoted_user, quoted_username
    - URLs: urls, entities_urls, links, expanded_urls
    - Hashtags: hashtags, entities_hashtags

    **Response Format**:
    ```json
    {
      "success": true,
      "edges": {
        "mention_edges": [{"src": "user1", "dst": "user2"}, ...],
        "reply_edges": [...],
        "retweet_edges": [...],
        "quote_edges": [...],
        "domain_edges": [{"src": "user1", "dst": "example.com"}, ...],
        "hashtag_edges": [{"src": "user1", "dst": "#politics"}, ...],
        "nodes": [{"id": "user1", "type": "user"}, {"id": "example.com", "type": "domain"}, ...]
      },
      "stats": {
        "mention_count": 150,
        "reply_count": 50,
        "retweet_count": 30,
        "quote_count": 10,
        "domain_count": 80,
        "hashtag_count": 40,
        "node_count": 200
      }
    }
    ```

    **Example Usage**:
    ```bash
    # CSV
    curl -F "file=@tweets.csv" http://localhost:8002/prepare

    # XLSX with hashtags
    curl -F "file=@data.xlsx" http://localhost:8002/prepare?sheet=0&extract_hashtags=true

    # Then use the edges for analysis
    # Save mention_edges to file and upload to /metrics endpoint
    ```
    """
    try:
        # Read file
        file_bytes = await file.read()
        filename = file.filename or "data.csv"

        # Load DataFrame
        if filename.lower().endswith(('.xlsx', '.xlsm', '.xls')):
            # Parse sheet parameter (could be int or string)
            try:
                sheet_param = int(sheet)
            except ValueError:
                sheet_param = sheet

            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_param)
        else:
            # Assume CSV
            df = pd.read_csv(io.BytesIO(file_bytes))

        # Extract edges
        edges = extract_social_media_edges(df, extract_hashtags=extract_hashtags)

        # Convert to dict
        result = edges.to_dict()

        return {
            "success": True,
            "edges": {
                "mention_edges": result["mention_edges"],
                "reply_edges": result["reply_edges"],
                "retweet_edges": result["retweet_edges"],
                "quote_edges": result["quote_edges"],
                "domain_edges": result["domain_edges"],
                "hashtag_edges": result["hashtag_edges"],
                "nodes": result["nodes"]
            },
            "stats": result["stats"]
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect required columns: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Social media graph preparation failed: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8002))

    print("=" * 70)
    print("CiceroWatch Graph Analytics Service")
    print("=" * 70)
    print(f"Listening on: http://0.0.0.0:{port}")
    print("Features: Similarity, Clustering, Networks, Communities")
    print("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
