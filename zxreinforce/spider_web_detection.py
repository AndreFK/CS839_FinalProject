"""
Spider-web detection for ZX diagrams.

Spider-webs are dense, cyclic clusters of nodes that are difficult to optimize.
This module provides functions to detect and measure spider-webs.
"""

import numpy as np
from collections import deque


def get_neighbours(node_idx, source, target):
    """Get all neighbors of a node"""
    neighbours = []
    for i in range(len(source)):
        if source[i] == node_idx:
            neighbours.append(target[i])
        elif target[i] == node_idx:
            neighbours.append(source[i])
    return np.array(neighbours, dtype=np.int32)


def detect_spider_webs(colors, source, target, min_cluster_size=3, min_density=0.5):
    """
    Detect spider-webs (dense clusters) in a ZX diagram.
    
    Args:
        colors: Array of node colors
        source: Array of edge sources
        target: Array of edge targets
        min_cluster_size: Minimum number of nodes in a cluster to be considered a spider-web
        min_density: Minimum edge density (edges / max_possible_edges) for a cluster
    
    Returns:
        dict with keys:
            'clusters': list of lists, each inner list contains node indices in a cluster
            'sizes': list of cluster sizes
            'densities': list of cluster densities
            'total_webs': int, total number of spider-webs detected
    """
    n_nodes = len(colors)
    if n_nodes == 0:
        return {'clusters': [], 'sizes': [], 'densities': [], 'total_webs': 0}
    
    # Build adjacency list
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(len(source)):
        s, t = source[i], target[i]
        adj_list[s].append(t)
        adj_list[t].append(s)
    
    # Find connected components using BFS
    visited = np.zeros(n_nodes, dtype=bool)
    clusters = []
    
    for start_node in range(n_nodes):
        if visited[start_node]:
            continue
        
        # BFS to find connected component
        cluster = []
        queue = deque([start_node])
        visited[start_node] = True
        
        while queue:
            node = queue.popleft()
            cluster.append(node)
            
            for neighbor in adj_list[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    
    # Calculate density for each cluster
    cluster_sizes = []
    cluster_densities = []
    spider_webs = []
    
    for cluster in clusters:
        cluster_size = len(cluster)
        cluster_sizes.append(cluster_size)
        
        # Count edges within cluster
        edges_in_cluster = 0
        cluster_set = set(cluster)
        
        for i in range(len(source)):
            s, t = source[i], target[i]
            if s in cluster_set and t in cluster_set:
                edges_in_cluster += 1
        
        # Maximum possible edges in cluster (complete graph)
        max_edges = cluster_size * (cluster_size - 1) // 2
        density = edges_in_cluster / max_edges if max_edges > 0 else 0.0
        
        cluster_densities.append(density)
        
        # Consider it a spider-web if density is high enough
        if density >= min_density:
            spider_webs.append(cluster)
    
    return {
        'clusters': clusters,
        'sizes': cluster_sizes,
        'densities': cluster_densities,
        'spider_webs': spider_webs,
        'total_webs': len(spider_webs)
    }


def count_spider_web_nodes(colors, source, target, min_cluster_size=3, min_density=0.5):
    """
    Count the number of nodes that are part of spider-webs.
    
    Returns:
        int: Number of nodes in spider-webs
    """
    webs = detect_spider_webs(colors, source, target, min_cluster_size, min_density)
    web_nodes = set()
    for cluster in webs['spider_webs']:
        web_nodes.update(cluster)
    return len(web_nodes)


def get_spider_web_penalty(colors, source, target, min_cluster_size=3, min_density=0.5):
    """
    Calculate a penalty score for spider-webs.
    Higher score means more/bigger spider-webs.
    
    Returns:
        float: Penalty score (0.0 = no spider-webs, higher = worse)
    """
    webs = detect_spider_webs(colors, source, target, min_cluster_size, min_density)
    
    if webs['total_webs'] == 0:
        return 0.0
    
    # Penalty based on number of webs and their sizes
    total_web_nodes = sum(len(cluster) for cluster in webs['spider_webs'])
    n_nodes = len(colors)
    
    # Normalize by total number of nodes
    penalty = total_web_nodes / n_nodes if n_nodes > 0 else 0.0
    
    return penalty

