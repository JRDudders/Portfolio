"""
Graph Analytics Microservice

Handles all graph-related tasks:
- Text similarity analysis
- Document clustering
- Network graph generation
- Community detection
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os

# Import graph processing modules
from graph_processor import (
    compute_similarity_matrix,
    perform_clustering,
    build_network_graph,
    detect_communities
)

app = FastAPI(
    title="CiceroWatch Graph Analytics Service",
    description="Graph analytics and clustering microservice",
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
# Request Models
# ============================================================

class SimilarityRequest(BaseModel):
    texts: List[str]
    method: str = "cosine"  # cosine, jaccard, etc.


class ClusteringRequest(BaseModel):
    texts: List[str]
    num_clusters: Optional[int] = None
    method: str = "kmeans"  # kmeans, hierarchical, dbscan


class NetworkRequest(BaseModel):
    texts: List[str]
    threshold: float = 0.5  # similarity threshold


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
# Graph Analytics Endpoints
# ============================================================

@app.post("/similarity")
async def calculate_similarity(request: SimilarityRequest):
    """
    Calculate similarity matrix for texts
    """
    try:
        if len(request.texts) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 texts for similarity analysis"
            )

        result = compute_similarity_matrix(
            texts=request.texts,
            method=request.method
        )

        return {
            "success": True,
            "num_texts": len(request.texts),
            "method": request.method,
            "similarity_matrix": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cluster")
async def cluster_texts(request: ClusteringRequest):
    """
    Cluster texts using specified method
    """
    try:
        if len(request.texts) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 texts for clustering"
            )

        result = perform_clustering(
            texts=request.texts,
            num_clusters=request.num_clusters,
            method=request.method
        )

        return {
            "success": True,
            "num_texts": len(request.texts),
            "method": request.method,
            "clusters": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/network")
async def build_network(request: NetworkRequest):
    """
    Build network graph from text similarities
    """
    try:
        if len(request.texts) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 texts for network analysis"
            )

        result = build_network_graph(
            texts=request.texts,
            threshold=request.threshold
        )

        return {
            "success": True,
            "num_texts": len(request.texts),
            "threshold": request.threshold,
            "network": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/communities")
async def find_communities(request: NetworkRequest):
    """
    Detect communities in text network
    """
    try:
        if len(request.texts) < 3:
            raise HTTPException(
                status_code=400,
                detail="Need at least 3 texts for community detection"
            )

        result = detect_communities(
            texts=request.texts,
            threshold=request.threshold
        )

        return {
            "success": True,
            "num_texts": len(request.texts),
            "communities": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
