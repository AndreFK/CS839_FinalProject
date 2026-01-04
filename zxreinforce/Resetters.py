# Holds resetters for the ZX env returning 
# (colors, angles, selected_node, source, target, selected_edges)

import numpy as np
from .own_constants import (INPUT, OUTPUT, GREEN, RED, HADAMARD, 
                      ZERO, PI_half, PI, PI_three_half, ARBITRARY, NO_ANGLE,
                      ANGLE_LIST)
from .ZX_env import apply_auto_actions, add_edge, remove_edge, breadth_first_search, get_neighbours
from .flow_checking import has_flow


def ensure_outputs_reachable_from_inputs(colors, source, target, input_indices, output_indices, rng):
    """
    Ensure each output is reachable from at least one input.
    If an output is not reachable, add a path from a reachable node to that output.
    Note: This may add an edge to an output, but we'll fix the "exactly one edge" constraint afterwards.
    """
    n_nodes = len(colors)
    
    # Find all nodes reachable from any input
    all_reachable_from_inputs = set()
    for inp_idx in input_indices:
        reachable = breadth_first_search(inp_idx, source, target, n_nodes)
        all_reachable_from_inputs.update(reachable)
    
    # Check each output and fix if not reachable
    for out_idx in output_indices:
        if out_idx not in all_reachable_from_inputs:
            # Output is not reachable from any input - need to add a path
            # Check if output already has an incoming edge
            output_edge_indices = np.where(target == out_idx)[0]
            output_has_edge = len(output_edge_indices) > 0
            
            # Find spider nodes (non-input, non-output) that are reachable from inputs
            spider_indices = np.where(np.logical_and(
                np.logical_not(np.all(colors == INPUT, axis=1)),
                np.logical_not(np.all(colors == OUTPUT, axis=1))
            ))[0]
            
            reachable_spiders = [s for s in spider_indices if s in all_reachable_from_inputs]
            
            if len(reachable_spiders) > 0:
                # Prefer connecting from a reachable spider to the unreachable output
                # If output already has an edge, we'll need to remove it first to maintain "exactly one edge" constraint
                if output_has_edge:
                    # Remove the existing edge (the one that doesn't connect to reachable nodes)
                    for edge_idx in sorted(output_edge_indices, reverse=True):
                        source, target = remove_edge(edge_idx, source, target)
                source_node = rng.choice(reachable_spiders)
                source, target = add_edge(source_node, out_idx, source, target)
                # Update reachable set
                new_reachable = breadth_first_search(source_node, source, target, n_nodes)
                all_reachable_from_inputs.update(new_reachable)
            elif len(input_indices) > 0:
                # If no reachable spiders, connect directly from an input
                # If output already has an edge, remove it first
                if output_has_edge:
                    for edge_idx in sorted(output_edge_indices, reverse=True):
                        source, target = remove_edge(edge_idx, source, target)
                source_node = rng.choice(input_indices)
                source, target = add_edge(source_node, out_idx, source, target)
                # Update reachable set
                new_reachable = breadth_first_search(source_node, source, target, n_nodes)
                all_reachable_from_inputs.update(new_reachable)
    
    return source, target


def ensure_at_least_one_input_output(colors, angles, source, target, rng):
    """
    Ensure the diagram has at least one input and one output node.
    If missing, add them and connect them appropriately.
    """
    input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
    output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
    
    # Find spider nodes (non-input, non-output)
    spider_indices = np.where(np.logical_and(
        np.logical_not(np.all(colors == INPUT, axis=1)),
        np.logical_not(np.all(colors == OUTPUT, axis=1))
    ))[0]
    
    # Add input if missing
    if len(input_indices) == 0:
        # Add an input node
        colors = np.row_stack((colors, INPUT))
        angles = np.row_stack((angles, NO_ANGLE))
        input_idx = len(colors) - 1
        
        # Connect input to a spider node if available, or to output if no spiders
        if len(spider_indices) > 0:
            target_node = rng.choice(spider_indices)
            source, target = add_edge(input_idx, target_node, source, target)
        elif len(output_indices) > 0:
            # Connect directly to an output
            target_node = rng.choice(output_indices)
            source, target = add_edge(input_idx, target_node, source, target)
        # If no spiders and no outputs, we'll add an output next
    
    # Add output if missing
    if len(output_indices) == 0:
        # Add an output node
        colors = np.row_stack((colors, OUTPUT))
        angles = np.row_stack((angles, NO_ANGLE))
        output_idx = len(colors) - 1
        
        # Re-find spider indices (in case input was just added)
        spider_indices = np.where(np.logical_and(
            np.logical_not(np.all(colors == INPUT, axis=1)),
            np.logical_not(np.all(colors == OUTPUT, axis=1))
        ))[0]
        
        # Connect output from a spider node if available, or from input if no spiders
        if len(spider_indices) > 0:
            source_node = rng.choice(spider_indices)
            source, target = add_edge(source_node, output_idx, source, target)
        else:
            # Connect directly from an input
            input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
            if len(input_indices) > 0:
                source_node = rng.choice(input_indices)
                source, target = add_edge(source_node, output_idx, source, target)
    
    return colors, angles, source, target


def ensure_flow(colors, source, target, rng, max_attempts=5):
    """
    Ensure the diagram has flow. If not, try to fix it by removing problematic edges.
    
    Args:
        colors: Array of node colors
        source: Array of edge sources
        target: Array of edge targets
        rng: Random number generator
        max_attempts: Maximum number of attempts to fix flow
    
    Returns:
        tuple: (colors, source, target, has_flow_flag)
            has_flow_flag: True if flow was achieved, False otherwise
    """
    from .flow_checking import has_flow as check_flow
    
    # Check if flow exists
    if check_flow(colors, source, target, INPUT, OUTPUT):
        return colors, source, target, True
    
    # Try to fix flow by removing backward/same-layer edges
    input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
    output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
    
    if len(input_indices) == 0 or len(output_indices) == 0:
        # Can't fix flow without inputs/outputs
        return colors, source, target, False
    
    # Build adjacency list and compute layers
    n_nodes = len(colors)
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(len(source)):
        s, t = source[i], target[i]
        adj_list[s].append(t)
        adj_list[t].append(s)
    
    # Compute layers using BFS from inputs
    layers = {}
    from collections import deque
    queue = deque(input_indices)
    visited = set(input_indices)
    
    for inp in input_indices:
        layers[inp] = 0
    
    while queue:
        node = queue.popleft()
        current_layer = layers[node]
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                layers[neighbor] = current_layer + 1
                queue.append(neighbor)
            elif neighbor in layers:
                layers[neighbor] = min(layers[neighbor], current_layer + 1)
    
    # Find problematic edges (backward or same-layer)
    # We need to remove edges one at a time and rebuild the graph structure
    # to avoid index issues when edges are removed
    
    max_iterations = 20  # Prevent infinite loops
    iteration = 0
    edges_removed = 0
    max_edges_to_remove = max(1, len(source) // 2)  # Don't remove more than half the edges
    
    while iteration < max_iterations and edges_removed < max_edges_to_remove:
        iteration += 1
        
        # Rebuild adjacency list and layers after each removal
        n_nodes = len(colors)
        adj_list = [[] for _ in range(n_nodes)]
        for i in range(len(source)):
            s, t = source[i], target[i]
            adj_list[s].append(t)
            adj_list[t].append(s)
        
        # Recompute layers
        layers = {}
        queue = deque(input_indices)
        visited = set(input_indices)
        for inp in input_indices:
            layers[inp] = 0
        
        while queue:
            node = queue.popleft()
            current_layer = layers[node]
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    layers[neighbor] = current_layer + 1
                    queue.append(neighbor)
        
        # Find problematic edges (using current valid indices)
        backward_edges_list = []
        same_layer_edges_list = []
        
        for i in range(len(source)):  # Use current length of source
            s, t = source[i], target[i]
            s_layer = layers.get(s, -1)
            t_layer = layers.get(t, -1)
            if s_layer >= 0 and t_layer >= 0:
                if s_layer > t_layer:
                    backward_edges_list.append(i)
                elif s_layer == t_layer and s != t:
                    same_layer_edges_list.append(i)
        
        # Remove one backward edge if available (prioritize backward edges)
        if len(backward_edges_list) > 0:
            edge_idx = rng.choice(backward_edges_list)
            source, target = remove_edge(edge_idx, source, target)
            edges_removed += 1
            continue
        
        # Remove one same-layer edge if available
        if len(same_layer_edges_list) > 0:
            edge_idx = rng.choice(same_layer_edges_list)
            source, target = remove_edge(edge_idx, source, target)
            edges_removed += 1
            continue
        
        # No more problematic edges to remove
        break
    
    # Check flow after removing problematic edges
    if check_flow(colors, source, target, INPUT, OUTPUT):
        return colors, source, target, True
    
    # If still no flow, the issue might be that we removed too many edges
    # or the structure is fundamentally broken. Try to ensure at least connectivity
    # by checking if outputs are still reachable
    input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
    output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
    
    # Check if outputs are reachable
    all_reachable = set()
    for inp in input_indices:
        reachable = breadth_first_search(inp, source, target, len(colors))
        all_reachable.update(reachable)
    
    # If outputs are not reachable, we need to add forward edges
    for out_idx in output_indices:
        if out_idx not in all_reachable:
            # Find a node that is reachable and at a lower layer than output
            # Add a forward edge from that node to the output
            reachable_spiders = [n for n in all_reachable 
                               if not np.all(colors[n] == INPUT) and 
                               not np.all(colors[n] == OUTPUT)]
            if len(reachable_spiders) > 0:
                # Choose a node that's likely at a lower layer
                source_node = rng.choice(reachable_spiders)
                source, target = add_edge(source_node, out_idx, source, target)
                # Update reachable set
                new_reachable = breadth_first_search(source_node, source, target, len(colors))
                all_reachable.update(new_reachable)
    
    # Final check
    has_flow_result = check_flow(colors, source, target, INPUT, OUTPUT)
    return colors, source, target, has_flow_result


class Resetter_ZERO_PI_PIHALF_ARB_hada():
    def __init__(self,
                 n_in_min:int,
                 n_in_max:int,
                 min_spiders:int,
                 max_spiders:int,
                 pi_fac:float,
                 pi_half_fac:float,
                 arb_fac:float,
                 p_hada:float,
                 min_mean_neighbours:int,
                 max_mean_neighbours:int,
                 rng:np.random.Generator):
        """n_in_min: minimum number of input spiders,
        n_in_max: maximum number of input spiders,
        min_spiders: minimum number of spiders in total,
        max_spiders: maximum number of spiders in total,
        pi_fac: factor by which to reduce probability of pi angle,
        pi_half_fac: factor by which to reduce probability of pi/2 angle,
        arb_fac: factor by which to reduce probability of arbitrary angle,
        p_hada: factor by which to reduce probability of hadamard node,
        min_mean_neighbours: minimum number of neighbours per node,
        max_mean_neighbours: maximum number of neighbours per node,
        rng: numpy random generator"""
        self.n_in_min = n_in_min
        self.n_in_max = n_in_max
        self.min_spiders = min_spiders
        self.max_spiders = max_spiders
        self.pi_fac = pi_fac
        self.pi_half_fac = pi_half_fac
        self.arb_fac = arb_fac
        self.p_hada = p_hada
        self.min_mean_neighbours = min_mean_neighbours
        self.max_mean_neighbours = max_mean_neighbours
        self.rng = rng

    def reset(self)->tuple:
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Builds random ZX diagrams"""
        # Sample inout and output number unifomrly
        n_input = self.rng.integers(low=self.n_in_min, high=self.n_in_max+1)
        n_output = self.rng.integers(low=self.n_in_min, high=self.n_in_max+1)
        # Sample number of spiders uniformly
        n_init_spiders = self.rng.integers(low=self.min_spiders, high=self.max_spiders+1)
        # Sample number of hadamards
        n_hada  = self.rng.integers(low=0, high=(n_init_spiders * self.p_hada))
        # Make sure there is at least one spider
        n_init_spiders = np.max([n_init_spiders, 1])

        # Sample neighbour number uniformly
        mean_neighbours = self.rng.integers(low=self.min_mean_neighbours, high=self.max_mean_neighbours+1)
        # Calculate probability of each edge such that self.mean_neighbours 
        # is expected value of neighbours per node
        p_edge = (mean_neighbours - (n_input + n_output) / n_init_spiders) / (n_init_spiders+1)
        if p_edge < 0:
            p_edge = 0

        # Sample probabilities for angles uniformly, reduce, and normalize
        p_zero, p_pi, p_pi_half, p_arb = self.rng.uniform(size=4)
        p_pi *= self.pi_fac
        p_arb *= self.arb_fac
        p_pi_half *= self.pi_half_fac

        norm = p_zero + p_pi + p_arb + p_pi_half
        p_zero /= norm
        p_pi /= norm
        p_arb /= norm
        p_pi_half /= norm

        ps_angle = [0]*(len(ZERO)-1)
        ps_angle[np.where(PI)[0][0]] = p_pi
        ps_angle[np.where(PI_half)[0][0]] = p_pi_half
        ps_angle[np.where(ZERO)[0][0]] = p_zero
        ps_angle[np.where(ARBITRARY)[0][0]] = p_arb

        # Sample color of each spider randomly
        red_spiders= self.rng.binomial(n_init_spiders, 0.5)
        # Hackky way to make 1d numpy array out of angle list
        angle_arr = np.empty(len(ANGLE_LIST)+1, dtype="O")
        angle_arr[:] = (ANGLE_LIST+[ARBITRARY])[:]

        # Create angle list for all spiders
        node_angle_list = self.rng.choice(angle_arr, size=n_init_spiders, p=ps_angle) 
        node_angle_list = node_angle_list

        # Create color list for all spiders
        node_color_list = np.array([RED] * red_spiders + [GREEN] * (n_init_spiders - red_spiders))
        self.rng.shuffle(node_color_list, axis=0)
        node_color_list = node_color_list.tolist()

        colors = np.array([INPUT] * n_input + [OUTPUT] * n_output + node_color_list)
        angles = np.array([NO_ANGLE] * (n_input + n_output) + list(node_angle_list))

        # Create edges
        edge_source, edge_target = np.where(np.triu(np.reshape(
            self.rng.choice(2, size=int(n_init_spiders**2), p=(1-p_edge, p_edge)), (n_init_spiders, n_init_spiders)), k=1))
        
        source = np.array(list(np.arange(n_input + n_output)) + list(edge_source + n_input + n_output))
        target = np.array(list(np.arange(n_input + n_output, 2 * n_input + n_output)) + 
                       list(np.arange(2 * n_input + n_output, 2 * n_input + 2 * n_output)) + 
                       list(edge_target + n_input + n_output))

        # Apply automatic actions
        colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
        
        # Add hadamards
        idcs_to_connect = np.arange(n_input + n_output, len(colors))

        if len(idcs_to_connect) >= 2:
            for _ in range(n_hada):

                idx_new_node = len(colors)
                colors = np.row_stack((colors, HADAMARD))
                angles = np.row_stack((angles, NO_ANGLE))


                connected_idx1 = self.rng.choice(idcs_to_connect, 1)[0]
                new_to_connect = np.delete(idcs_to_connect, np.where(idcs_to_connect==connected_idx1)[0])
                connected_idx2 = self.rng.choice(new_to_connect, 1)[0]

                source, target = add_edge(idx_new_node, connected_idx1, source, target)
                source, target = add_edge(idx_new_node, connected_idx2, source, target)

        # Apply automatic actions
        colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
        
        # After apply_auto_actions, some nodes might have been removed
        # Ensure inputs and outputs still have exactly one edge
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Check and fix inputs
        for inp_idx in input_indices:
            input_edge_count = np.sum(source == inp_idx)
            if input_edge_count == 0:
                # Input lost its edge, reconnect to a random spider node
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    target_node = self.rng.choice(spider_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
            elif input_edge_count > 1:
                # Input has multiple edges, keep only the first one
                edge_indices = np.where(source == inp_idx)[0]
                # Remove in reverse order to maintain indices
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Check and fix outputs
        for out_idx in output_indices:
            output_edge_count = np.sum(target == out_idx)
            if output_edge_count == 0:
                # Output lost its edge, reconnect from a random spider node
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    source_node = self.rng.choice(spider_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
            elif output_edge_count > 1:
                # Output has multiple edges, keep only the first one
                edge_indices = np.where(target == out_idx)[0]
                # Remove in reverse order to maintain indices
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Ensure each output is reachable from at least one input (for flow preservation)
        source, target = ensure_outputs_reachable_from_inputs(
            colors, source, target, input_indices, output_indices, self.rng)
        
        # Ensure at least one input and one output exist
        colors, angles, source, target = ensure_at_least_one_input_output(
            colors, angles, source, target, self.rng)
        
        # After adding inputs/outputs, re-check and fix edge constraints
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Ensure inputs have exactly one outgoing edge
        for inp_idx in input_indices:
            input_edge_count = np.sum(source == inp_idx)
            if input_edge_count == 0:
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    target_node = self.rng.choice(spider_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
                elif len(output_indices) > 0:
                    target_node = self.rng.choice(output_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
            elif input_edge_count > 1:
                edge_indices = np.where(source == inp_idx)[0]
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Ensure outputs have exactly one incoming edge
        for out_idx in output_indices:
            output_edge_count = np.sum(target == out_idx)
            if output_edge_count == 0:
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    source_node = self.rng.choice(spider_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
                elif len(input_indices) > 0:
                    source_node = self.rng.choice(input_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
            elif output_edge_count > 1:
                edge_indices = np.where(target == out_idx)[0]
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Final check: ensure outputs are still reachable from inputs
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        source, target = ensure_outputs_reachable_from_inputs(
            colors, source, target, input_indices, output_indices, self.rng)
        
        # Final validation: ensure at least one input and one output exist
        colors, angles, source, target = ensure_at_least_one_input_output(
            colors, angles, source, target, self.rng)
        
        # Verify inputs/outputs still exist (safety check)
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Debug: Print warning if inputs/outputs are missing (should never happen)
        if len(input_indices) == 0:
            print(f"WARNING: Resetter_ZERO_PI_PIHALF_ARB_hada generated diagram with NO INPUTS! Adding input...")
        if len(output_indices) == 0:
            print(f"WARNING: Resetter_ZERO_PI_PIHALF_ARB_hada generated diagram with NO OUTPUTS! Adding output...")
        
        if len(input_indices) == 0 or len(output_indices) == 0:
            # If still missing, force add them
            if len(input_indices) == 0:
                colors = np.row_stack((colors, INPUT))
                angles = np.row_stack((angles, NO_ANGLE))
                input_idx = len(colors) - 1
                if len(output_indices) > 0:
                    source, target = add_edge(input_idx, output_indices[0], source, target)
                elif len(colors) > 1:
                    spider_idx = len(colors) - 2
                    source, target = add_edge(input_idx, spider_idx, source, target)
            if len(output_indices) == 0:
                colors = np.row_stack((colors, OUTPUT))
                angles = np.row_stack((angles, NO_ANGLE))
                output_idx = len(colors) - 1
                input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
                if len(input_indices) > 0:
                    source, target = add_edge(input_indices[0], output_idx, source, target)
                elif len(colors) > 1:
                    spider_idx = len(colors) - 2
                    source, target = add_edge(spider_idx, output_idx, source, target)
        
        # Final verification
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        assert len(input_indices) > 0, "Resetter_ZERO_PI_PIHALF_ARB_hada: No inputs after all fixes!"
        assert len(output_indices) > 0, "Resetter_ZERO_PI_PIHALF_ARB_hada: No outputs after all fixes!"
        
        # Ensure flow exists
        colors, source, target, has_flow_result = ensure_flow(colors, source, target, self.rng)
        if not has_flow_result:
            print(f"WARNING: Resetter_ZERO_PI_PIHALF_ARB_hada generated diagram without flow after fixes. "
                  f"Diagram may still be invalid.")
        
        # Re-check edge constraints after flow fix (flow fix might have removed edges)
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Ensure inputs have exactly one outgoing edge
        for inp_idx in input_indices:
            input_edge_count = np.sum(source == inp_idx)
            if input_edge_count == 0:
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    target_node = self.rng.choice(spider_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
                elif len(output_indices) > 0:
                    target_node = self.rng.choice(output_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
            elif input_edge_count > 1:
                edge_indices = np.where(source == inp_idx)[0]
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Ensure outputs have exactly one incoming edge
        for out_idx in output_indices:
            output_edge_count = np.sum(target == out_idx)
            if output_edge_count == 0:
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    source_node = self.rng.choice(spider_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
                elif len(input_indices) > 0:
                    source_node = self.rng.choice(input_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
            elif output_edge_count > 1:
                edge_indices = np.where(target == out_idx)[0]
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Check if diagram has flow using PyZX gflow algorithm
        # If it doesn't have flow, try to fix it by ensuring outputs are reachable
        if not has_flow(colors, source, target, INPUT, OUTPUT, angles=angles):
            # Try to ensure outputs are reachable from inputs
            input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
            output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
            
            if len(input_indices) > 0 and len(output_indices) > 0:
                # Find all nodes reachable from inputs
                all_reachable = set()
                for inp in input_indices:
                    reachable = breadth_first_search(inp, source, target, len(colors))
                    all_reachable.update(reachable)
                
                # Connect unreachable outputs to reachable nodes
                for out_idx in output_indices:
                    if out_idx not in all_reachable:
                        # Find reachable spider nodes
                        reachable_spiders = [n for n in all_reachable 
                                           if not np.all(colors[n] == INPUT) and 
                                           not np.all(colors[n] == OUTPUT)]
                        if len(reachable_spiders) > 0:
                            source_node = self.rng.choice(reachable_spiders)
                            source, target = add_edge(source_node, out_idx, source, target)
                            # Update reachable set
                            new_reachable = breadth_first_search(source_node, source, target, len(colors))
                            all_reachable.update(new_reachable)
        
        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)

        return colors, angles, selected_node, source, target, selected_edges

class Resetter_FlowPatterns():
    """
    Resetter that creates ZX diagrams with flow (gFlow/cFlow).
    
    Creates diagrams in a layered structure (inputs -> layers -> outputs)
    to ensure a DAG structure that typically has flow.
    """
    def __init__(self,
                 n_in_min: int,
                 n_in_max: int,
                 n_out_min: int,
                 n_out_max: int,
                 n_layers_min: int,
                 n_layers_max: int,
                 nodes_per_layer_min: int,
                 nodes_per_layer_max: int,
                 p_edge: float,
                 pi_fac: float,
                 pi_half_fac: float,
                 arb_fac: float,
                 p_hada: float,
                 rng: np.random.Generator):
        """
        n_in_min: minimum number of input nodes
        n_in_max: maximum number of input nodes
        n_out_min: minimum number of output nodes
        n_out_max: maximum number of output nodes
        n_layers_min: minimum number of intermediate layers
        n_layers_max: maximum number of intermediate layers
        nodes_per_layer_min: minimum nodes per intermediate layer
        nodes_per_layer_max: maximum nodes per intermediate layer
        p_edge: probability of edge between consecutive layers
        pi_fac: factor to reduce probability of pi angle
        pi_half_fac: factor to reduce probability of pi/2 angle
        arb_fac: factor to reduce probability of arbitrary angle
        p_hada: probability of hadamard node
        rng: numpy random generator
        """
        self.n_in_min = n_in_min
        self.n_in_max = n_in_max
        self.n_out_min = n_out_min
        self.n_out_max = n_out_max
        self.n_layers_min = n_layers_min
        self.n_layers_max = n_layers_max
        self.nodes_per_layer_min = nodes_per_layer_min
        self.nodes_per_layer_max = nodes_per_layer_max
        self.p_edge = p_edge
        self.pi_fac = pi_fac
        self.pi_half_fac = pi_half_fac
        self.arb_fac = arb_fac
        self.p_hada = p_hada
        self.rng = rng

    def reset(self) -> tuple:
        """
        Returns (colors, angles, selected_node, source, target, selected_edges)
        Builds a layered ZX diagram that should have flow.
        """
        # Sample number of inputs and outputs
        n_input = self.rng.integers(low=self.n_in_min, high=self.n_in_max + 1)
        n_output = self.rng.integers(low=self.n_out_min, high=self.n_out_max + 1)
        
        # Sample number of intermediate layers
        n_layers = self.rng.integers(low=self.n_layers_min, high=self.n_layers_max + 1)
        
        # Sample number of nodes per layer
        layer_sizes = [self.rng.integers(low=self.nodes_per_layer_min, 
                                         high=self.nodes_per_layer_max + 1) 
                       for _ in range(n_layers)]
        
        # Build node indices: inputs, then layers, then outputs
        node_idx = 0
        input_start = node_idx
        input_end = node_idx + n_input
        node_idx = input_end
        
        layer_starts = []
        layer_ends = []
        for layer_size in layer_sizes:
            layer_starts.append(node_idx)
            node_idx += layer_size
            layer_ends.append(node_idx)
        
        output_start = node_idx
        output_end = node_idx + n_output
        total_nodes = output_end
        
        # Initialize colors and angles
        colors = []
        angles = []
        
        # Input nodes
        colors.extend([INPUT] * n_input)
        angles.extend([NO_ANGLE] * n_input)
        
        # Intermediate layer nodes
        for layer_size in layer_sizes:
            # Sample angles
            p_zero, p_pi, p_pi_half, p_arb = self.rng.uniform(size=4)
            p_pi *= self.pi_fac
            p_arb *= self.arb_fac
            p_pi_half *= self.pi_half_fac
            
            norm = p_zero + p_pi + p_arb + p_pi_half
            if norm > 0:
                p_zero /= norm
                p_pi /= norm
                p_arb /= norm
                p_pi_half /= norm
            
            # Sample colors (red or green)
            red_count = self.rng.binomial(layer_size, 0.5)
            layer_colors = [RED] * red_count + [GREEN] * (layer_size - red_count)
            self.rng.shuffle(layer_colors)
            colors.extend(layer_colors)
            
            # Sample angles
            angle_arr = np.empty(len(ANGLE_LIST) + 1, dtype="O")
            angle_arr[:] = (ANGLE_LIST + [ARBITRARY])[:]
            ps_angle = [0] * (len(ZERO) - 1)
            ps_angle[np.where(PI)[0][0]] = p_pi
            ps_angle[np.where(PI_half)[0][0]] = p_pi_half
            ps_angle[np.where(ZERO)[0][0]] = p_zero
            ps_angle[np.where(ARBITRARY)[0][0]] = p_arb
            
            layer_angles = self.rng.choice(angle_arr, size=layer_size, p=ps_angle)
            angles.extend(layer_angles.tolist())
        
        # Output nodes
        colors.extend([OUTPUT] * n_output)
        angles.extend([NO_ANGLE] * n_output)
        
        colors = np.array(colors)
        angles = np.array(angles)
        
        # Build edges in a layered fashion (only forward edges)
        # IMPORTANT: Each input must have exactly one outgoing edge
        #            Each output must have exactly one incoming edge
        source = []
        target = []
        
        if n_layers > 0:
            # Edges from inputs to first layer
            # Each input connects to exactly one node in the first layer
            first_layer_nodes = list(range(layer_starts[0], layer_ends[0]))
            for input_idx in range(input_start, input_end):
                # Randomly select one target from first layer
                target_node = self.rng.choice(first_layer_nodes)
                source.append(input_idx)
                target.append(target_node)
            
            # Edges between consecutive layers
            for layer_i in range(n_layers - 1):
                for src_idx in range(layer_starts[layer_i], layer_ends[layer_i]):
                    for tgt_idx in range(layer_starts[layer_i + 1], layer_ends[layer_i + 1]):
                        if self.rng.random() < self.p_edge:
                            source.append(src_idx)
                            target.append(tgt_idx)
            
            # Edges from last layer to outputs
            # Each output receives exactly one connection from the last layer
            last_layer_nodes = list(range(layer_starts[-1], layer_ends[-1]))
            for output_idx in range(output_start, output_end):
                # Randomly select one source from last layer
                source_node = self.rng.choice(last_layer_nodes)
                source.append(source_node)
                target.append(output_idx)
        else:
            # If no intermediate layers, connect inputs directly to outputs
            # Each input-output pair gets exactly one edge
            # We need n_input >= n_output for this to work, or we pair them up
            output_nodes = list(range(output_start, output_end))
            for i, input_idx in enumerate(range(input_start, input_end)):
                # Pair each input with an output (cycling if needed)
                output_idx = output_nodes[i % len(output_nodes)]
                source.append(input_idx)
                target.append(output_idx)
        
        source = np.array(source, dtype=np.int32)
        target = np.array(target, dtype=np.int32)
        
        # Apply automatic actions (removes identity spiders, etc.)
        colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
        
        # Add some hadamards randomly
        spider_indices = np.arange(len(colors))
        # Filter out inputs and outputs
        spider_mask = np.logical_not(
            np.logical_or(np.all(colors == INPUT, axis=1), 
                         np.all(colors == OUTPUT, axis=1)))
        spider_indices = spider_indices[spider_mask]
        
        n_hada = self.rng.binomial(len(spider_indices), self.p_hada)
        hada_indices = self.rng.choice(spider_indices, size=min(n_hada, len(spider_indices)), 
                                       replace=False)
        
        for hada_idx in hada_indices:
            # Convert to hadamard
            colors[hada_idx] = HADAMARD
            angles[hada_idx] = NO_ANGLE
        
        # Apply automatic actions again after adding hadamards
        colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
        
        # After apply_auto_actions, some nodes might have been removed
        # Ensure inputs and outputs still have exactly one edge
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Check and fix inputs
        for inp_idx in input_indices:
            input_edge_count = np.sum(source == inp_idx)
            if input_edge_count == 0:
                # Input lost its edge, reconnect to a random spider node
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    target_node = self.rng.choice(spider_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
            elif input_edge_count > 1:
                # Input has multiple edges, keep only the first one
                edge_indices = np.where(source == inp_idx)[0]
                # Remove in reverse order to maintain indices
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Check and fix outputs
        for out_idx in output_indices:
            output_edge_count = np.sum(target == out_idx)
            if output_edge_count == 0:
                # Output lost its edge, reconnect from a random spider node
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    source_node = self.rng.choice(spider_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
            elif output_edge_count > 1:
                # Output has multiple edges, keep only the first one
                edge_indices = np.where(target == out_idx)[0]
                # Remove in reverse order to maintain indices
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Ensure each output is reachable from at least one input (for flow preservation)
        source, target = ensure_outputs_reachable_from_inputs(
            colors, source, target, input_indices, output_indices, self.rng)
        
        # Ensure at least one input and one output exist
        colors, angles, source, target = ensure_at_least_one_input_output(
            colors, angles, source, target, self.rng)
        
        # After adding inputs/outputs, re-check and fix edge constraints
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Ensure inputs have exactly one outgoing edge
        for inp_idx in input_indices:
            input_edge_count = np.sum(source == inp_idx)
            if input_edge_count == 0:
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    target_node = self.rng.choice(spider_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
                elif len(output_indices) > 0:
                    target_node = self.rng.choice(output_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
            elif input_edge_count > 1:
                edge_indices = np.where(source == inp_idx)[0]
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Ensure outputs have exactly one incoming edge
        for out_idx in output_indices:
            output_edge_count = np.sum(target == out_idx)
            if output_edge_count == 0:
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    source_node = self.rng.choice(spider_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
                elif len(input_indices) > 0:
                    source_node = self.rng.choice(input_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
            elif output_edge_count > 1:
                edge_indices = np.where(target == out_idx)[0]
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Final check: ensure outputs are still reachable from inputs
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        source, target = ensure_outputs_reachable_from_inputs(
            colors, source, target, input_indices, output_indices, self.rng)
        
        # Final validation: ensure at least one input and one output exist
        colors, angles, source, target = ensure_at_least_one_input_output(
            colors, angles, source, target, self.rng)
        
        # Verify inputs/outputs still exist (safety check)
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Debug: Print warning if inputs/outputs are missing (should never happen)
        if len(input_indices) == 0:
            print(f"WARNING: Resetter_FlowPatterns generated diagram with NO INPUTS! Adding input...")
        if len(output_indices) == 0:
            print(f"WARNING: Resetter_FlowPatterns generated diagram with NO OUTPUTS! Adding output...")
        
        if len(input_indices) == 0 or len(output_indices) == 0:
            # If still missing, force add them
            if len(input_indices) == 0:
                colors = np.row_stack((colors, INPUT))
                angles = np.row_stack((angles, NO_ANGLE))
                input_idx = len(colors) - 1
                if len(output_indices) > 0:
                    source, target = add_edge(input_idx, output_indices[0], source, target)
                elif len(colors) > 1:
                    spider_idx = len(colors) - 2
                    source, target = add_edge(input_idx, spider_idx, source, target)
            if len(output_indices) == 0:
                colors = np.row_stack((colors, OUTPUT))
                angles = np.row_stack((angles, NO_ANGLE))
                output_idx = len(colors) - 1
                input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
                if len(input_indices) > 0:
                    source, target = add_edge(input_indices[0], output_idx, source, target)
                elif len(colors) > 1:
                    spider_idx = len(colors) - 2
                    source, target = add_edge(spider_idx, output_idx, source, target)
        
        # Final verification
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        assert len(input_indices) > 0, "Resetter_FlowPatterns: No inputs after all fixes!"
        assert len(output_indices) > 0, "Resetter_FlowPatterns: No outputs after all fixes!"
        
        # Ensure flow exists (FlowPatterns should have flow, but check after auto_actions)
        colors, source, target, has_flow_result = ensure_flow(colors, source, target, self.rng)
        if not has_flow_result:
            print(f"WARNING: Resetter_FlowPatterns generated diagram without flow after fixes. "
                  f"This should not happen for FlowPatterns resetter.")
        
        # Re-check edge constraints after flow fix (flow fix might have removed edges)
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Ensure inputs have exactly one outgoing edge
        for inp_idx in input_indices:
            input_edge_count = np.sum(source == inp_idx)
            if input_edge_count == 0:
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    target_node = self.rng.choice(spider_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
                elif len(output_indices) > 0:
                    target_node = self.rng.choice(output_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
            elif input_edge_count > 1:
                edge_indices = np.where(source == inp_idx)[0]
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        # Ensure outputs have exactly one incoming edge
        for out_idx in output_indices:
            output_edge_count = np.sum(target == out_idx)
            if output_edge_count == 0:
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    source_node = self.rng.choice(spider_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
                elif len(input_indices) > 0:
                    source_node = self.rng.choice(input_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
            elif output_edge_count > 1:
                edge_indices = np.where(target == out_idx)[0]
                for edge_idx in sorted(edge_indices[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
        
        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)
        
        return colors, angles, selected_node, source, target, selected_edges


class Resetter_BrickworkPattern():
    """
    Resetter that creates brickwork/cluster-style ZX diagrams with obvious flow.
    
    Creates a grid structure where:
    - Qubits = rows
    - Time steps = columns
    - Inputs = left column
    - Outputs = right column
    
    This structure naturally has flow due to its DAG-like grid topology.
    """
    def __init__(self,
                 n_qubits_min: int,
                 n_qubits_max: int,
                 depth_min: int,
                 depth_max: int,
                 rng: np.random.Generator):
        """
        n_qubits_min: minimum number of qubits (rows)
        n_qubits_max: maximum number of qubits (rows)
        depth_min: minimum depth (number of time steps/columns)
        depth_max: maximum depth (number of time steps/columns)
        rng: numpy random generator
        """
        self.n_qubits_min = n_qubits_min
        self.n_qubits_max = n_qubits_max
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.rng = rng

    def reset(self) -> tuple:
        """
        Returns (colors, angles, selected_node, source, target, selected_edges)
        Builds a brickwork/cluster-style ZX diagram with flow.
        """
        # Sample number of qubits (rows) and depth (columns)
        n = int(self.rng.integers(self.n_qubits_min, self.n_qubits_max + 1))
        d = int(self.rng.integers(self.depth_min, self.depth_max + 1))
        
        # Total nodes: n qubits × (d+1) time steps
        # (d+1 because we have d+1 columns: inputs + d intermediate + outputs)
        num_nodes = n * (d + 1)
        
        # Start with all GREEN spiders with ZERO angle
        colors = np.array([GREEN] * num_nodes)
        angles = np.array([ZERO] * num_nodes)
        
        # Define inputs and outputs (by index) BEFORE creating edges
        # Inputs: left column (t=0 for each qubit)
        inputs = [i * (d + 1) + 0 for i in range(n)]
        # Outputs: right column (t=d for each qubit)
        outputs = [i * (d + 1) + d for i in range(n)]
        
        # Relabel inputs and outputs
        for v in inputs:
            colors[v] = INPUT
            angles[v] = NO_ANGLE
        for v in outputs:
            colors[v] = OUTPUT
            angles[v] = NO_ANGLE
        
        # Build edges: horizontal (time direction) + vertical (between neighboring qubits)
        # IMPORTANT: Don't create edges between inputs or between outputs
        src = []
        tgt = []
        
        for i in range(n):  # For each qubit (row)
            for t in range(d + 1):  # For each time step (column)
                v = i * (d + 1) + t  # Node index
                
                # Skip if this is an input or output (we'll handle them separately)
                is_input = v in inputs
                is_output = v in outputs
                
                # Horizontal CZ (time direction): connect to next time step
                # Only create if source is not an output and target is not an input
                if t < d:
                    w = i * (d + 1) + (t + 1)
                    # Don't create edge if connecting input to input or output to output
                    w_is_input = w in inputs
                    w_is_output = w in outputs
                    if not ((is_input and w_is_input) or (is_output and w_is_output)):
                        src.append(v)
                        tgt.append(w)
                
                # Vertical CZ (between neighboring qubits): connect to qubit below
                # Only create if not connecting inputs to inputs or outputs to outputs
                if i < n - 1:
                    w = (i + 1) * (d + 1) + t
                    w_is_input = w in inputs
                    w_is_output = w in outputs
                    # Don't create edge if connecting input to input or output to output
                    if not ((is_input and w_is_input) or (is_output and w_is_output)):
                        src.append(v)
                        tgt.append(w)
        
        source = np.array(src, dtype=np.int32)
        target = np.array(tgt, dtype=np.int32)
        
        # Apply automatic actions (removes identity spiders, etc.)
        colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target
        )
        
        # After apply_auto_actions, enforce edge constraints for inputs/outputs
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Remove any edges that connect inputs to inputs or outputs to outputs
        edges_to_remove = []
        for i in range(len(source)):
            s, t = source[i], target[i]
            s_is_input = s in input_indices
            s_is_output = s in output_indices
            t_is_input = t in input_indices
            t_is_output = t in output_indices
            
            # Remove edges between inputs or between outputs
            if (s_is_input and t_is_input) or (s_is_output and t_is_output):
                edges_to_remove.append(i)
        
        # Remove edges in reverse order to maintain indices
        for edge_idx in sorted(edges_to_remove, reverse=True):
            source, target = remove_edge(edge_idx, source, target)
        
        # Recompute indices after edge removal
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Ensure inputs have exactly one outgoing edge
        for inp_idx in input_indices:
            # Count outgoing edges (where input is source)
            outgoing_edges = np.where(source == inp_idx)[0]
            if len(outgoing_edges) == 0:
                # No outgoing edge - add one to a spider node or output
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    target_node = self.rng.choice(spider_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
                elif len(output_indices) > 0:
                    target_node = self.rng.choice(output_indices)
                    source, target = add_edge(inp_idx, target_node, source, target)
            elif len(outgoing_edges) > 1:
                # Multiple outgoing edges - keep only the first one
                for edge_idx in sorted(outgoing_edges[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
            
            # Remove any incoming edges to inputs (inputs should only have outgoing edges)
            incoming_edges = np.where(target == inp_idx)[0]
            for edge_idx in sorted(incoming_edges, reverse=True):
                source, target = remove_edge(edge_idx, source, target)
        
        # Ensure outputs have exactly one incoming edge
        for out_idx in output_indices:
            # Count incoming edges (where output is target)
            incoming_edges = np.where(target == out_idx)[0]
            if len(incoming_edges) == 0:
                # No incoming edge - add one from a spider node or input
                spider_indices = np.where(np.logical_and(
                    np.logical_not(np.all(colors == INPUT, axis=1)),
                    np.logical_not(np.all(colors == OUTPUT, axis=1))
                ))[0]
                if len(spider_indices) > 0:
                    source_node = self.rng.choice(spider_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
                elif len(input_indices) > 0:
                    source_node = self.rng.choice(input_indices)
                    source, target = add_edge(source_node, out_idx, source, target)
            elif len(incoming_edges) > 1:
                # Multiple incoming edges - keep only the first one
                for edge_idx in sorted(incoming_edges[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
            
            # Remove any outgoing edges from outputs (outputs should only have incoming edges)
            outgoing_edges = np.where(source == out_idx)[0]
            for edge_idx in sorted(outgoing_edges, reverse=True):
                source, target = remove_edge(edge_idx, source, target)
        
        # Final check: ensure outputs are reachable from inputs
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        source, target = ensure_outputs_reachable_from_inputs(
            colors, source, target, input_indices, output_indices, self.rng)
        
        # After ensuring reachability, re-enforce edge constraints (in case ensure_outputs_reachable added edges)
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Ensure inputs have exactly one outgoing edge (again, after reachability fixes)
        for inp_idx in input_indices:
            outgoing_edges = np.where(source == inp_idx)[0]
            if len(outgoing_edges) > 1:
                for edge_idx in sorted(outgoing_edges[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
            incoming_edges = np.where(target == inp_idx)[0]
            for edge_idx in sorted(incoming_edges, reverse=True):
                source, target = remove_edge(edge_idx, source, target)
        
        # Ensure outputs have exactly one incoming edge (again, after reachability fixes)
        for out_idx in output_indices:
            incoming_edges = np.where(target == out_idx)[0]
            if len(incoming_edges) > 1:
                for edge_idx in sorted(incoming_edges[1:], reverse=True):
                    source, target = remove_edge(edge_idx, source, target)
            outgoing_edges = np.where(source == out_idx)[0]
            for edge_idx in sorted(outgoing_edges, reverse=True):
                source, target = remove_edge(edge_idx, source, target)
        
        # Check if diagram has flow using PyZX gflow algorithm
        # Brickwork patterns should naturally have flow, but check after all fixes
        if not has_flow(colors, source, target, INPUT, OUTPUT, angles=angles):
            # Try to ensure outputs are reachable from inputs
            input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
            output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
            
            if len(input_indices) > 0 and len(output_indices) > 0:
                # Find all nodes reachable from inputs
                all_reachable = set()
                for inp in input_indices:
                    reachable = breadth_first_search(inp, source, target, len(colors))
                    all_reachable.update(reachable)
                
                # Connect unreachable outputs to reachable nodes
                for out_idx in output_indices:
                    if out_idx not in all_reachable:
                        # Find reachable spider nodes
                        reachable_spiders = [n for n in all_reachable 
                                           if not np.all(colors[n] == INPUT) and 
                                           not np.all(colors[n] == OUTPUT)]
                        if len(reachable_spiders) > 0:
                            source_node = self.rng.choice(reachable_spiders)
                            # Remove existing incoming edge to output first (if any)
                            existing_incoming = np.where(target == out_idx)[0]
                            for edge_idx in sorted(existing_incoming, reverse=True):
                                source, target = remove_edge(edge_idx, source, target)
                            source, target = add_edge(source_node, out_idx, source, target)
                            # Update reachable set
                            new_reachable = breadth_first_search(source_node, source, target, len(colors))
                            all_reachable.update(new_reachable)
        
        # Final verification: ensure edge constraints are still satisfied
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        # Verify inputs have exactly one outgoing edge and no incoming edges
        for inp_idx in input_indices:
            outgoing_count = np.sum(source == inp_idx)
            incoming_count = np.sum(target == inp_idx)
            assert outgoing_count == 1, f"Input {inp_idx} has {outgoing_count} outgoing edges (should be 1)"
            assert incoming_count == 0, f"Input {inp_idx} has {incoming_count} incoming edges (should be 0)"
        
        # Verify outputs have exactly one incoming edge and no outgoing edges
        for out_idx in output_indices:
            incoming_count = np.sum(target == out_idx)
            outgoing_count = np.sum(source == out_idx)
            assert incoming_count == 1, f"Output {out_idx} has {incoming_count} incoming edges (should be 1)"
            assert outgoing_count == 0, f"Output {out_idx} has {outgoing_count} outgoing edges (should be 0)"
        
        selected_node = np.zeros(len(colors), dtype=np.int32)
        selected_edges = np.zeros(len(source), dtype=np.int32)
        
        return colors, angles, selected_node, source, target, selected_edges

class Resetter_Test_COPY():
    def __init__(self, 
                 n_out:int,
                 n_extra_node:int, 
                 angle_copy:list=ZERO, 
                 color_zero:list=GREEN, 
                 angles_out:list=ARBITRARY, 
                 angle_other:list=ARBITRARY):
        """n_out: number of output nodes,
        n_extra_node: number of extra nodes inserted on outputs,
        angle_copy: angle of copy node,
        color_zero: color of zero node,
        angles_out: angle of output nodes,
        angle_other: angle of extra nodes"""
        self.n_out = n_out
        self.n_extra_node = n_extra_node
        self.color_zero = color_zero
        self.angle_copy=angle_copy
        # Spiders need oposite color
        if np.all(self.color_zero == GREEN):
            self.other_col = RED
        else:
            self.other_col = GREEN
        self.angles_out=angles_out
        self.angle_other=angle_other
    
    def reset(self):
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Builds Copy test ZX diagram"""
        colors = np.array([self.color_zero] * 1 + [self.other_col] * 1 
                            + [self.color_zero] * self.n_extra_node 
                            + [OUTPUT]* self.n_out)
        angles = np.array([self.angle_copy] * 1 + [self.angle_other] * 1 
                            + [self.angles_out] * self.n_extra_node
                            + [NO_ANGLE]* self.n_out)

        source = [0,]
        target = [1,]
        for i in range(self.n_extra_node):

            source.append(1)
            target.append(2 + i)

            source.append(2 + i)
            target.append(2 + i + self.n_extra_node)

        for i in range(self.n_extra_node, self.n_out):
            source.append(1)
            target.append(2 + i + self.n_extra_node)

        source = np.array(source, dtype=np.int32)
        target = np.array(target, dtype=np.int32)


        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)

        return colors, angles, selected_node, source, target, selected_edges
    

class ResetterBialgUnmerge():
    def reset(self)->tuple:
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Bialgabra diagram with one ectra output"""
        colors=[]
        angles=[]

        colors.append(INPUT)
        colors.append(INPUT)

        colors.append(GREEN)
        colors.append(GREEN)
        colors.append(RED)
        colors.append(RED)

        colors.append(OUTPUT)
        colors.append(OUTPUT)
        colors.append(OUTPUT)

        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)

        angles.append(ZERO)
        angles.append(ZERO)
        angles.append(ZERO)
        angles.append(ZERO)

        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)

        source = [0, 1, 2, 2, 3, 3, 4, 5, 5,]
        target = [2, 3, 4, 5, 4, 5, 6, 7, 8,]

        selected_node = [0]*len(colors)
        selected_edges = [0]*len(source)

        return (np.array(colors), np.array(angles), np.array(selected_node),
                np.array(source), np.array(target), np.array(selected_edges))
    


class ResetterBialgLvl2():
    def reset(self)->tuple:
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Bialgabra diagram level 2"""
        colors=[]
        angles=[]

        colors.append(INPUT)
        colors.append(INPUT)

        colors.append(GREEN)
        colors.append(GREEN)
        colors.append(RED)
        colors.append(RED)
        colors.append(RED)

        colors.append(OUTPUT)
        colors.append(OUTPUT)
        colors.append(OUTPUT)

        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)

        angles.append(ZERO)
        angles.append(ZERO)
        angles.append(ZERO)
        angles.append(ZERO)
        angles.append(ZERO)

        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)

        source = [0, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6,]
        target = [2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9,]

        selected_node = [0]*len(colors)
        selected_edges = [0]*len(source)

        return (np.array(colors), np.array(angles), np.array(selected_node),
                np.array(source), np.array(target), np.array(selected_edges))




        

class Bilagebra_lvl_n_m():
    def __init__(self, 
                 n_min:int,
                 n_max:int,
                 rng:np.random.Generator):
        """n_min: minimum number of nodes on one side,
        n_max: maximum number of nodes on one side
        rng: numpy random generator"""
        self.n_min = n_min
        self.n_max = n_max
        self.rng=rng


    def reset(self):
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Builds Bialgebra with n_min to n_max nodes on each side"""
        n_green = self.rng.integers(low=self.n_min, high=self.n_max+1)
        n_red = self.rng.integers(low=self.n_min, high=self.n_max+1)

        colors = np.array(
            [INPUT] * n_green + [GREEN, ] * n_green + [RED,] * n_red + [OUTPUT,] * n_red, 
            dtype=np.int32
        )
        angles = np.array(
            [NO_ANGLE] * n_green + [ZERO, ] * n_green + [ZERO,] * n_red + [NO_ANGLE,] * n_red, 
            dtype=np.int32
        )

        source = []
        target = []
        # Edges between input and green
        for i in range(n_green):
            source.append(i)
            target.append(n_green + i)
        # Edges between red and output
        for i in range(2*n_green, 2*n_green + n_red):
            source.append(i)
            target.append(n_red + i)
        # Edges between green and red
        for i in range(n_green, 2*n_green):
            for j in range(2*n_green, 2*n_green + n_red):
                source.append(i)
                target.append(j)

        source = np.array(source, dtype=np.int32)
        target = np.array(target, dtype=np.int32)


        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)

        return colors, angles, selected_node, source, target, selected_edges
    

class Bilagebra_add_edges():
    def __init__(self, 
                 add_min:int,
                 add_max:int,
                 rng:np.random.Generator):
        """add_min: minimum number of edges to add,
        add_max: maximum number of edges to add
        rng: numpy random generator"""
        self.add_min = add_min
        self.add_max = add_max
        self.rng=rng


    def reset(self):
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Builds Bialgebra multiple outputs added to one node"""
        add_max = self.rng.integers(low=self.add_min, high=self.add_max+1)
        index_conn = self.rng.integers(low=0, high=4)
        print(index_conn)

        source = []
        target = []

        if index_conn < 2:
            n_input = 2 + add_max
            n_output = 2
            for i in range(2, 2 + add_max):
                source.append(index_conn+n_input)
                target.append(i)
        else:
            n_input = 2
            n_output = 2 + add_max
            for i in range(n_input + 6, n_input + 6 + add_max):
                source.append(index_conn+n_input)
                target.append(i)

        # Edges between green and red
        for i in range(n_input, n_input+2):
            source.append(i)
            target.append(n_input+2)
            source.append(i)
            target.append(n_input+3)

        # At least on edge between input and green
        for i in range(0, 2):
            source.append(i)
            target.append(n_input + i)

        # At least on edge between output and red
        for i in range(n_input + 2, n_input + 4):
            source.append(i)
            target.append(2 + i)

        colors = np.array(
            [INPUT] * n_input + [GREEN, ] * 2 + [RED,] * 2 + [OUTPUT,] * n_output, 
            dtype=np.int32
        )
        angles = np.array(
            [NO_ANGLE] * n_input + [ZERO, ] * 2 + [ZERO,] * 2 + [NO_ANGLE,] * n_output, 
            dtype=np.int32
        )


        source = np.array(source, dtype=np.int32)
        target = np.array(target, dtype=np.int32)


        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)

        return colors, angles, selected_node, source, target, selected_edges
    


class Bilagebra_add_angle():
    def __init__(self, 
                 rng:np.random.Generator):
        """rng: numpy random generator"""
        self.rng=rng


    def reset(self):
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Builds Bialgebra with random angle added to one node"""
        index_angle = self.rng.integers(low=0, high=4)
        angle_type = self.rng.choice([ZERO, PI_half, PI, PI_three_half, ARBITRARY])

        source = []
        target = []

        if index_angle < 2 :
            n_input = 1
            n_output = 2

            source.append(0)
            target.append(2 - index_angle)

            source.append(3)
            target.append(5)
            source.append(4)
            target.append(6)
        else:
            n_input = 2
            n_output = 1

            source.append(0)
            target.append(2)
            source.append(1)
            target.append(3)

            source.append(6)
            target.append(7 - index_angle)


        # Edges between green and red
        for i in range(n_input, n_input+2):
            source.append(i)
            target.append(n_input+2)
            source.append(i)
            target.append(n_input+3)

        colors = np.array(
            [INPUT] * n_input + [GREEN, ] * 2 + [RED,] * 2 + [OUTPUT,] * n_output, 
            dtype=np.int32
        )
        angles = np.array(
            [NO_ANGLE] * n_input + [ZERO, ] * 2 + [ZERO,] * 2 + [NO_ANGLE,] * n_output, 
            dtype=np.int32
        )

        angles[n_input+index_angle] = angle_type


        source = np.array(source, dtype=np.int32)
        target = np.array(target, dtype=np.int32)


        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)

        return colors, angles, selected_node, source, target, selected_edges

