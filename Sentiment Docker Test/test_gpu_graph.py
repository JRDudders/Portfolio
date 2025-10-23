#!/usr/bin/env python
"""
GPU Graph Analytics Test
Tests GPU acceleration with a sample graph and monitors GPU usage.
"""

import time
import sys

print("=" * 70)
print("GPU GRAPH ANALYTICS TEST")
print("=" * 70)

# Step 1: Import and check GPU detection
print("\n1. Importing graph_tasks and checking GPU detection...")
try:
    import graph_tasks
    print(f"   HAS_GPU: {graph_tasks.HAS_GPU}")
    print(f"   HAS_CUGRAPH: {graph_tasks.HAS_CUGRAPH}")

    if graph_tasks.HAS_GPU:
        print("   ✅ GPU DETECTED - Will use GPU acceleration")
    else:
        print("   ⚠️  No GPU - Will use CPU")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Step 2: Create a test graph
print("\n2. Creating test graph (1000 nodes, ~5000 edges)...")
import io
edges = []
for i in range(1000):
    # Create edges to form a network
    edges.append(f"{i},{(i+1)%1000}")
    edges.append(f"{i},{(i+2)%1000}")
    edges.append(f"{i},{(i+5)%1000}")
    edges.append(f"{i},{(i+10)%1000}")
    edges.append(f"{i},{(i+50)%1000}")

edge_csv = "src,dst\n" + "\n".join(edges)
print(f"   Created graph with {len(edges)} edges")

# Step 3: Load graph
print("\n3. Loading graph...")
start = time.time()
g = graph_tasks.load_graph_from_bytes(edge_csv.encode(), 'csv')
load_time = time.time() - start
print(f"   ✅ Graph loaded: {g.n} nodes, {len(g.edges)} edges")
print(f"   Time: {load_time:.3f}s")

# Step 4: Run degrees (GPU test)
print("\n4. Computing degrees...")
print("   👀 WATCH GPU USAGE NOW (nvidia-smi in another terminal)")
start = time.time()
degrees = graph_tasks.degrees(g)
degrees_time = time.time() - start
print(f"   ✅ Degrees computed")
print(f"   Time: {degrees_time:.3f}s")
print(f"   Sample: {degrees.head(3).to_dict('records')}")

# Step 5: Run PageRank (GPU test)
print("\n5. Computing PageRank...")
print("   👀 WATCH GPU USAGE NOW")
start = time.time()
pr = graph_tasks.pagerank(g, iters=50)
pr_time = time.time() - start
print(f"   ✅ PageRank computed")
print(f"   Time: {pr_time:.3f}s")
print(f"   Sample: {pr.head(3).to_dict('records')}")

# Step 6: Run BFS (GPU test)
print("\n6. Computing BFS from node 0...")
print("   👀 WATCH GPU USAGE NOW")
start = time.time()
bfs = graph_tasks.bfs(g, "0")
bfs_time = time.time() - start
print(f"   ✅ BFS computed")
print(f"   Time: {bfs_time:.3f}s")
print(f"   Max distance: {bfs['distance'].max()}")

# Step 7: Run triangles (GPU test)
print("\n7. Computing triangles...")
print("   👀 WATCH GPU USAGE NOW")
start = time.time()
triangles = graph_tasks.triangles_undirected(g)
tri_time = time.time() - start
print(f"   ✅ Triangles computed")
print(f"   Time: {tri_time:.3f}s")
print(f"   Result: {triangles}")

# Summary
print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)
print(f"Graph loading:     {load_time:.3f}s")
print(f"Degrees:           {degrees_time:.3f}s")
print(f"PageRank (50 it):  {pr_time:.3f}s")
print(f"BFS:               {bfs_time:.3f}s")
print(f"Triangles:         {tri_time:.3f}s")
print(f"\nTotal compute:     {degrees_time + pr_time + bfs_time + tri_time:.3f}s")

if graph_tasks.HAS_GPU:
    print("\n✅ GPU acceleration is ACTIVE")
    print("   These times should be 10-100x faster than CPU")
    print("\n💡 Monitor GPU usage with: nvidia-smi -l 1")
else:
    print("\n⚠️  Running on CPU (no GPU detected)")
    print("   GPU version would be 10-100x faster")

print("=" * 70)
