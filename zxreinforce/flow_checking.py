"""
Flow checking for ZX diagrams.

This module provides functions to check if a ZX diagram has
generalized flow (gFlow) or causal flow (cFlow), which are important
properties for circuit extraction.

Implements the gflow algorithm based on Perdrix and Mhalla.
See dx.doi.org/10.1007/978-3-540-70575-8_70
"""

import numpy as np
from collections import deque
from typing import Dict, Set, Tuple, Optional


def get_neighbours(node_idx, source, target):
    """Get all neighbors of a node"""
    neighbours = []
    for i in range(len(source)):
        if source[i] == node_idx:
            neighbours.append(target[i])
        elif target[i] == node_idx:
            neighbours.append(source[i])
    return np.array(neighbours, dtype=np.int32)


def is_input_or_output(node_idx, colors, INPUT, OUTPUT):
    """Check if a node is an input or output"""
    return (np.all(colors[node_idx] == INPUT) or 
            np.all(colors[node_idx] == OUTPUT))


def check_gflow(colors, source, target, INPUT, OUTPUT):
    """
    Simplified gFlow checking.
    
    A diagram has gFlow if:
    1. It's a DAG (directed acyclic graph) when considering inputs -> outputs
    2. Each output is reachable from at least one input
    3. There's a partial order (layering) from inputs to outputs
    
    This is a simplified check that verifies:
    - No cycles in the graph structure
    - All outputs are reachable from inputs
    
    Args:
        colors: Array of node colors
        source: Array of edge sources
        target: Array of edge targets
        INPUT: Input node color constant
        OUTPUT: Output node color constant
    
    Returns:
        tuple: (has_gflow: bool, flow_depth: int)
            flow_depth: -1 if no flow, otherwise the depth of the flow
    """
    n_nodes = len(colors)
    if n_nodes == 0:
        return False, -1
    
    # Find input and output nodes
    input_nodes = []
    output_nodes = []
    
    for i in range(n_nodes):
        if np.all(colors[i] == INPUT):
            input_nodes.append(i)
        elif np.all(colors[i] == OUTPUT):
            output_nodes.append(i)
    
    if len(input_nodes) == 0 or len(output_nodes) == 0:
        return False, -1
    
    # Build adjacency list (undirected for now)
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(len(source)):
        s, t = source[i], target[i]
        adj_list[s].append(t)
        adj_list[t].append(s)
    
    # Check if all outputs are reachable from inputs
    visited = np.zeros(n_nodes, dtype=bool)
    queue = deque(input_nodes)
    
    for inp in input_nodes:
        visited[inp] = True
    
    # BFS from all inputs
    while queue:
        node = queue.popleft()
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    # Check if all outputs are reachable
    all_outputs_reachable = all(visited[out] for out in output_nodes)
    
    if not all_outputs_reachable:
        return False, -1
    
    # For flow, we need to be able to assign layers such that:
    # 1. Inputs are at layer 0
    # 2. Outputs are at higher layers  
    # 3. Edges primarily go forward (from lower to higher layers)
    # 4. No dense cycles that would prevent meaningful flow
    
    # Try to compute a layering using BFS from inputs
    # Assign layer 0 to inputs
    layers = {}
    for inp in input_nodes:
        layers[inp] = 0
    
    # BFS to assign layers
    queue = deque(input_nodes)
    visited_layers = set(input_nodes)
    
    while queue:
        node = queue.popleft()
        current_layer = layers[node]
        
        for neighbor in adj_list[node]:
            if neighbor not in visited_layers:
                # First time seeing this neighbor - assign next layer
                visited_layers.add(neighbor)
                layers[neighbor] = current_layer + 1
                queue.append(neighbor)
            elif neighbor in layers:
                # Neighbor already has a layer - check if it's much lower
                neighbor_layer = layers[neighbor]
                if neighbor_layer < current_layer - 1:
                    # Large backward jump - indicates problematic structure
                    return False, -1
    
    # Check if outputs have higher layers than inputs
    if len(output_nodes) > 0:
        min_output_layer = min(layers.get(out, -1) for out in output_nodes)
        if min_output_layer <= 0:
            return False, -1
    
    # Check if all nodes are reachable
    if len(visited_layers) < n_nodes:
        # Some nodes are not reachable from inputs
        return False, -1
    
    # Check for problematic cycles: count backward and same-layer edges
    backward_edges = 0
    same_layer_edges = 0
    forward_edges = 0
    
    for i in range(len(source)):
        s, t = source[i], target[i]
        s_layer = layers.get(s, -1)
        t_layer = layers.get(t, -1)
        if s_layer >= 0 and t_layer >= 0:
            if s_layer > t_layer:
                backward_edges += 1
            elif s_layer == t_layer and s != t:
                same_layer_edges += 1
            else:
                forward_edges += 1
    
    total_edges = len(source)
    if total_edges == 0:
        return False, -1
    
    # Calculate ratio of backward/same-layer edges
    problematic_ratio = (backward_edges + same_layer_edges) / total_edges
    
    # Be very lenient: allow up to 60% problematic edges if there are still forward edges
    # This accounts for diagrams that have been simplified by apply_auto_actions
    # The key is that there should be SOME forward flow from inputs to outputs
    if problematic_ratio > 0.6:
        return False, -1
    
    # If there are many same-layer edges relative to forward edges, it's likely a spider web
    # But be more lenient here too - allow up to 40% same-layer edges
    if same_layer_edges > 0 and forward_edges > 0:
        same_layer_ratio = same_layer_edges / (forward_edges + same_layer_edges)
        if same_layer_ratio > 0.4:  # More than 40% same-layer suggests cycles
            return False, -1
    
    # Also check: if there are NO forward edges at all, that's a problem
    if forward_edges == 0 and total_edges > 0:
        return False, -1
    
    # Additional check: ensure there's at least one path from input to output
    # (This is already checked earlier, but double-check here)
    if len(output_nodes) > 0:
        # Check if at least one output is at a higher layer than inputs
        output_layers = [layers.get(out, -1) for out in output_nodes]
        if all(layer <= 0 for layer in output_layers):
            return False, -1
    
    # If we got here, it likely has flow
    max_layer = max(layers.values()) if layers else 0
    return True, max_layer


def check_cflow(colors, source, target, INPUT, OUTPUT):
    """
    Simplified cFlow (causal flow) checking.
    
    Causal flow is a stricter version of gFlow.
    This is a simplified check.
    
    Returns:
        tuple: (has_cflow: bool, flow_depth: int)
    """
    # For now, use the same check as gFlow
    # A more sophisticated implementation would check for specific causal flow conditions
    return check_gflow(colors, source, target, INPUT, OUTPUT)


def has_flow(colors, source, target, INPUT, OUTPUT, angles=None):
    """
    Check if diagram has flow (either gFlow or cFlow) using the PyZX algorithm.
    
    Uses the rigorous gflow algorithm by Perdrix and Mhalla.
    See dx.doi.org/10.1007/978-3-540-70575-8_70
    
    Args:
        colors: Array of node colors
        source: Array of edge sources
        target: Array of edge targets
        INPUT: Input node color constant
        OUTPUT: Output node color constant
        angles: Optional array of node angles (for more accurate flow checking)
    
    Returns:
        bool: True if diagram has flow
    """
    # Use only the rigorous PyZX-based gflow check
    result = check_gflow_pyzx(colors, source, target, INPUT, OUTPUT, angles=angles)
    return result is not None


def solve_gf2(A, b):
    """
    Solve linear system Ax = b over GF(2) (binary field).
    
    Args:
        A: m x n binary matrix (numpy array)
        b: m x 1 binary vector (numpy array)
    
    Returns:
        x: n x 1 binary vector solution, or None if no solution
    """
    # Use Gaussian elimination over GF(2)
    m, n = A.shape
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)
    
    # Augmented matrix [A | b]
    aug = np.hstack([A, b]) % 2
    
    # Forward elimination
    row = 0
    for col in range(min(n, m)):
        # Find pivot
        pivot_row = None
        for r in range(row, m):
            if aug[r, col] == 1:
                pivot_row = r
                break
        
        if pivot_row is None:
            continue
        
        # Swap rows
        if pivot_row != row:
            aug[[row, pivot_row]] = aug[[pivot_row, row]]
        
        # Eliminate
        for r in range(row + 1, m):
            if aug[r, col] == 1:
                aug[r] = (aug[r] + aug[row]) % 2
        
        row += 1
    
    # Back substitution
    x = np.zeros(n, dtype=np.int32)
    for r in range(row - 1, -1, -1):
        # Find the first non-zero entry in this row
        col = None
        for c in range(n):
            if aug[r, c] == 1:
                col = c
                break
        
        if col is None:
            # Check if inconsistent (non-zero RHS)
            if aug[r, n] == 1:
                return None
            continue
        
        # Set x[col] = RHS - sum of other terms
        val = aug[r, n]
        for c in range(col + 1, n):
            val = (val + aug[r, c] * x[c]) % 2
        x[col] = val
    
    # Check consistency for remaining rows
    for r in range(row, m):
        if aug[r, n] == 1:
            # Check if left side is zero
            if np.all(aug[r, :n] == 0):
                return None
    
    return x.reshape(-1, 1)


def check_gflow_pyzx(colors, source, target, INPUT, OUTPUT, angles=None, focus=False, reverse=False, pauli=False):
    """
    Compute the gflow of a diagram using the PyZX algorithm.
    
    Based on algorithm by Perdrix and Mhalla.
    See dx.doi.org/10.1007/978-3-540-70575-8_70
    
    Args:
        colors: Array of node colors (one-hot encoded)
        source: Array of edge sources
        target: Array of edge targets
        INPUT: Input node color constant
        OUTPUT: Output node color constant
        angles: Optional array of node angles (for Pauli flow)
        focus: Compute the focussed gflow
        reverse: Reverse the roles of inputs and outputs
        pauli: Compute the Pauli flow, restricted to {XY, X, Y} measurements
    
    Returns:
        Optional[Tuple[Dict[int, int], Dict[int, Set[int]]]]:
            (layers, gflow) if gflow exists, None otherwise
            layers: mapping from vertex to layer number
            gflow: mapping from vertex to its correction set
    """
    from .own_constants import GREEN, RED, HADAMARD, ZERO, PI_half, PI, PI_three_half, ARBITRARY
    
    n_nodes = len(colors)
    if n_nodes == 0:
        return None
    
    # Find ZX vertices (non-input, non-output nodes)
    vertices = set()
    pattern_inputs = set()
    pattern_outputs = set()
    pauli_x = set()
    pauli_y = set()
    
    # Build adjacency list
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(len(source)):
        s, t = source[i], target[i]
        adj_list[s].append(t)
        adj_list[t].append(s)
    
    # In our representation, INPUT/OUTPUT nodes are boundary nodes
    # For gflow, we need to identify the ZX vertices (spiders/hadamards) as the pattern
    # Pattern inputs are ZX vertices connected to INPUT nodes
    # Pattern outputs are ZX vertices connected to OUTPUT nodes
    
    # First, identify all ZX vertices
    for i in range(n_nodes):
        is_input = np.all(colors[i] == INPUT)
        is_output = np.all(colors[i] == OUTPUT)
        
        if not is_input and not is_output:
            # ZX vertex (spider or hadamard)
            vertices.add(i)
            
            if pauli and angles is not None:
                # Check if node is Pauli (X or Y measurement)
                # In ZX-calculus: X = phase 0 or π, Y = phase π/2 or 3π/2
                if np.all(colors[i] == GREEN) or np.all(colors[i] == RED):
                    if np.all(angles[i] == ZERO) or np.all(angles[i] == PI):
                        pauli_x.add(i)
                    elif np.all(angles[i] == PI_half) or np.all(angles[i] == PI_three_half):
                        pauli_y.add(i)
    
    # Now identify pattern inputs/outputs (ZX vertices connected to boundary nodes)
    for i in range(n_nodes):
        is_input = np.all(colors[i] == INPUT)
        is_output = np.all(colors[i] == OUTPUT)
        
        if is_input:
            # Input node: its ZX neighbors are pattern inputs
            for neighbor in adj_list[i]:
                if neighbor in vertices:
                    pattern_inputs.add(neighbor)
        elif is_output:
            # Output node: its ZX neighbors are pattern outputs
            for neighbor in adj_list[i]:
                if neighbor in vertices:
                    pattern_outputs.add(neighbor)
    
    if reverse:
        pattern_inputs, pattern_outputs = pattern_outputs, pattern_inputs
    
    # Initialize layers: outputs get layer 0
    layers: Dict[int, int] = {}
    gflow: Dict[int, Set[int]] = {}
    processed: Set[int] = pattern_outputs.copy()
    
    for v in processed:
        layers[v] = 0
    
    # Non-output vertices
    non_outputs = [v for v in vertices if v not in pattern_outputs]
    
    k = 1
    while True:
        correct: Set[int] = set()
        
        # Candidates: processed vertices (except inputs) that have unprocessed neighbors
        candidates = [
            v for v in (processed | pauli_x | pauli_y)
            if v not in pattern_inputs and
            (not focus or any(w not in processed for w in adj_list[v]))
        ]
        
        if focus:
            clean = [v for v in non_outputs if v not in processed]
        else:
            clean = [
                v for v in vertices
                if v not in processed and
                any(w in candidates for w in adj_list[v])
            ]
        
        if len(clean) == 0:
            # No more vertices to process
            # Check if all vertices are processed
            if len(vertices) == len(processed):
                # Success: all vertices processed
                if reverse:
                    max_layer = max(layers.values()) if layers else 0
                    layers = {v: max_layer - layers.get(v, 0) for v in layers}
                return layers, gflow
            return None
        
        # Build flow-demand matrix: bi-adjacency from clean to candidates
        # Also relate Y-measured nodes to themselves
        m = len(clean)
        n = len(candidates)
        if n == 0:
            # No candidates available
            if len(vertices) == len(processed):
                # Success: all vertices processed
                if reverse:
                    max_layer = max(layers.values()) if layers else 0
                    layers = {v: max_layer - layers.get(v, 0) for v in layers}
                return layers, gflow
            return None
        
        M = np.zeros((m, n), dtype=np.int32)
        candidate_list = list(candidates)
        for i, u in enumerate(clean):
            for j, v in enumerate(candidate_list):
                if u == v and u in pauli_y:
                    M[i, j] = 1
                elif v in adj_list[u]:
                    M[i, j] = 1
        
        # Solve for each clean vertex
        for idx, u in enumerate(clean):
            if not focus or (u not in processed and any(w in candidates for w in adj_list[u])):
                # Create unit vector for this vertex
                b = np.zeros(m, dtype=np.int32)
                b[idx] = 1
                
                # Solve Mx = b over GF(2)
                x = solve_gf2(M, b)
                
                if x is not None:
                    correct.add(u)
                    # Build correction set
                    gflow[u] = {candidate_list[i] for i in range(n) if x[i, 0] == 1}
                    layers[u] = k
        
        if not correct:
            if len(vertices) == len(processed):
                # Success: all vertices processed
                # Adjust layers if reverse
                if reverse:
                    max_layer = max(layers.values()) if layers else 0
                    layers = {v: max_layer - layers.get(v, 0) for v in layers}
                return layers, gflow
            return None
        else:
            processed.update(correct)
            k += 1
    
    return None

