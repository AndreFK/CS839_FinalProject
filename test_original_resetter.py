#!/usr/bin/env python3
"""
Script to visualize and test the Resetter_ZERO_PI_PIHALF_ARB_hada output.

This script generates a ZX diagram using Resetter_ZERO_PI_PIHALF_ARB_hada and displays:
- The diagram structure
- Statistics (nodes, edges, flow status, spider-webs)
- A visualization of the diagram
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

from zxreinforce.Resetters import Resetter_ZERO_PI_PIHALF_ARB_hada
from zxreinforce.ZX_env import ZXCalculus
from zxreinforce.plot_utils import plot_observation
from zxreinforce.flow_checking import check_gflow, has_flow, check_gflow_pyzx
from zxreinforce.spider_web_detection import detect_spider_webs, get_spider_web_penalty
from zxreinforce.own_constants import INPUT, OUTPUT


def print_diagram_stats(colors, angles, source, target):
    """Print statistics about the generated diagram"""
    n_nodes = len(colors)
    n_edges = len(source)
    
    # Count node types
    n_input = np.sum(np.all(colors == INPUT, axis=1))
    n_output = np.sum(np.all(colors == OUTPUT, axis=1))
    n_green = np.sum(np.all(colors == np.array([0,0,1,0,0]), axis=1))  # GREEN
    n_red = np.sum(np.all(colors == np.array([0,0,0,1,0]), axis=1))   # RED
    n_hadamard = np.sum(np.all(colors == np.array([0,0,0,0,1]), axis=1))  # HADAMARD
    
    # Check edge counts for inputs and outputs
    input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
    output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
    
    input_edge_counts = {}
    output_edge_counts = {}
    for i, (s, t) in enumerate(zip(source, target)):
        if s in input_indices:
            input_edge_counts[s] = input_edge_counts.get(s, 0) + 1
        if t in output_indices:
            output_edge_counts[t] = output_edge_counts.get(t, 0) + 1
    
    # Check flow using both methods
    has_gflow_simple, flow_depth = check_gflow(colors, source, target, INPUT, OUTPUT)
    has_gflow_pyzx = has_flow(colors, source, target, INPUT, OUTPUT, angles=angles)
    gflow_result_pyzx = check_gflow_pyzx(colors, source, target, INPUT, OUTPUT, angles=angles)
    
    # Check spider-webs
    webs = detect_spider_webs(colors, source, target)
    web_penalty = get_spider_web_penalty(colors, source, target)
    
    print("=" * 60)
    print("DIAGRAM STATISTICS")
    print("=" * 60)
    print(f"Total nodes: {n_nodes}")
    print(f"  - Inputs: {n_input}")
    print(f"  - Outputs: {n_output}")
    print(f"  - Green spiders: {n_green}")
    print(f"  - Red spiders: {n_red}")
    print(f"  - Hadamards: {n_hadamard}")
    print(f"\nTotal edges: {n_edges}")
    
    # Verify input/output edge constraints
    print(f"\nInput/Output Edge Verification:")
    all_inputs_ok = True
    for inp_idx in input_indices:
        count = input_edge_counts.get(inp_idx, 0)
        status = "✓" if count == 1 else "✗"
        if count != 1:
            all_inputs_ok = False
        print(f"  Input {inp_idx}: {count} edge(s) {status}")
    
    all_outputs_ok = True
    for out_idx in output_indices:
        count = output_edge_counts.get(out_idx, 0)
        status = "✓" if count == 1 else "✗"
        if count != 1:
            all_outputs_ok = False
        print(f"  Output {out_idx}: {count} edge(s) {status}")
    
    if not all_inputs_ok or not all_outputs_ok:
        print("  ⚠ WARNING: Some inputs/outputs don't have exactly one edge!")
    else:
        print("  ✓ All inputs/outputs have exactly one edge")
    
    print(f"\nFlow status:")
    print(f"  - Has gFlow (simplified): {has_gflow_simple}")
    print(f"  - Flow depth (simplified): {flow_depth}")
    print(f"  - Has flow (PyZX): {has_gflow_pyzx}")
    if gflow_result_pyzx is not None:
        layers, gflow = gflow_result_pyzx
        print(f"  - PyZX gFlow layers: {len(layers)} layers")
        print(f"  - PyZX gFlow assignments: {len(gflow)} corrections")
    else:
        print(f"  - PyZX gFlow: None (no flow detected)")
    
    print(f"\nSpider-webs:")
    print(f"  - Number of webs: {webs['total_webs']}")
    print(f"  - Web penalty: {web_penalty:.4f}")
    if webs['total_webs'] > 0:
        print(f"  - Web sizes: {[len(c) for c in webs['spider_webs']]}")
        print(f"  - Web densities: {[f'{d:.2f}' for d in webs['densities'][:len(webs['spider_webs'])]]}")
    print("=" * 60)


def main():
    """Main function to generate and visualize diagrams"""
    # Set random seed for reproducibility
    seed = 42
    rng = np.random.default_rng(seed)
    
    # Create resetter with reasonable parameters (similar to runner_final.py defaults)
    resetter = Resetter_ZERO_PI_PIHALF_ARB_hada(
        n_in_min=1,
        n_in_max=3,
        min_spiders=10,
        max_spiders=15,
        pi_fac=0.4,
        pi_half_fac=0.4,
        arb_fac=0.4,
        p_hada=0.2,
        min_mean_neighbours=2,
        max_mean_neighbours=4,
        rng=rng
    )
    
    print("Generating diagram with Resetter_ZERO_PI_PIHALF_ARB_hada...")
    print(f"Parameters:")
    print(f"  - Inputs: {resetter.n_in_min}-{resetter.n_in_max}")
    print(f"  - Outputs: {resetter.n_in_min}-{resetter.n_in_max}")
    print(f"  - Spiders: {resetter.min_spiders}-{resetter.max_spiders}")
    print(f"  - Mean neighbours: {resetter.min_mean_neighbours}-{resetter.max_mean_neighbours}")
    print(f"  - pi_fac: {resetter.pi_fac}")
    print(f"  - pi_half_fac: {resetter.pi_half_fac}")
    print(f"  - arb_fac: {resetter.arb_fac}")
    print(f"  - p_hada: {resetter.p_hada}")
    print()
    
    # Generate diagram
    colors, angles, selected_node, source, target, selected_edges = resetter.reset()
    
    # Print statistics
    print_diagram_stats(colors, angles, source, target)
    
    # Create environment to get observation format
    env = ZXCalculus(resetter=resetter, max_steps=1000)
    observation, mask = env.reset()
    
    # Plot the diagram
    print("\nPlotting diagram...")
    try:
        import networkx as nx
        from zxreinforce.plot_utils import observation_to_networkx, get_color_networkx, get_angle_networkx, get_color_edge_networkx
        import zxreinforce.own_constants as oc
        
        # Create a custom plot function that handles missing layer attributes
        colors, angles, selected_node, source, target, selected_edges, n_nodes, n_edges, context = observation
        
        # Build graph
        net_graph = observation_to_networkx(observation)
        
        # Try to assign layers, but use fallback if it fails
        try:
            from zxreinforce.plot_utils import sort_graph_into_layers, layer_list_to_node_attrs
            layer_list = sort_graph_into_layers(colors, source, target)
            layer_dict = layer_list_to_node_attrs(layer_list)
            nx.set_node_attributes(net_graph, layer_dict)
            
            # Check if all nodes have layer attribute
            all_nodes_have_layer = all('layer' in net_graph.nodes[node] for node in net_graph.nodes())
            if not all_nodes_have_layer:
                # Assign missing nodes to the last layer
                max_layer = max([net_graph.nodes[node].get('layer', 0) for node in net_graph.nodes()], default=0)
                for node in net_graph.nodes():
                    if 'layer' not in net_graph.nodes[node]:
                        net_graph.nodes[node]['layer'] = max_layer + 1
            
            pos = nx.multipartite_layout(net_graph, subset_key="layer")
        except Exception as layout_error:
            print(f"  Using fallback layout (multipartite failed: {layout_error})")
            # Use spring layout as fallback
            pos = nx.spring_layout(net_graph, k=2, iterations=50)
        
        # Get colors
        edge_colors = [get_color_edge_networkx(selected_edges[i]) for i in range(n_edges)]
        node_colors = [get_color_networkx(colors[i]) for i in range(n_nodes)]
        
        # Node labels
        node_labels = {}
        for i in range(n_nodes):
            angle_str = get_angle_networkx(angles[i])
            node_labels[i] = f"{i}\n{angle_str}"
        
        # Edge labels
        edge_labels = {}
        for i, (s, t) in enumerate(zip(source, target)):
            edge_labels[(s, t)] = i
        
        # Draw
        plt.figure(figsize=(14, 10))
        nx.draw(net_graph, pos=pos, labels=node_labels, edge_color=edge_colors, 
                node_color=node_colors, node_size=100, width=3, alpha=0.8)
        nx.draw_networkx_edge_labels(net_graph, pos, edge_labels=edge_labels, alpha=0.8,
                                     bbox={"boxstyle":'round', "ec":(1.0, 1.0, 1.0), 
                                           "fc":(1.0, 1.0, 1.0), "alpha":0.0})
        
        plt.title("ZX Diagram Generated by Resetter_ZERO_PI_PIHALF_ARB_hada", fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save plot to file
        output_file = "original_diagram_example.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Diagram saved to: {output_file}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot diagram: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with statistics only...")
    
    # Generate a few more examples
    print("\n" + "=" * 60)
    print("Generating 10 more examples for statistics...")
    print("=" * 60)
    
    flow_count_simple = 0
    flow_count_pyzx = 0
    total_webs = 0
    total_nodes = 0
    total_edges = 0
    total_inputs = 0
    total_outputs = 0
    edge_constraint_violations = 0
    
    for i in range(10):
        colors, angles, selected_node, source, target, selected_edges = resetter.reset()
        
        # Check flow
        has_gflow_simple, _ = check_gflow(colors, source, target, INPUT, OUTPUT)
        has_gflow_pyzx = has_flow(colors, source, target, INPUT, OUTPUT, angles=angles)
        
        # Check edge constraints
        input_indices = np.where(np.all(colors == INPUT, axis=1))[0]
        output_indices = np.where(np.all(colors == OUTPUT, axis=1))[0]
        
        input_edge_counts = {}
        output_edge_counts = {}
        for s, t in zip(source, target):
            if s in input_indices:
                input_edge_counts[s] = input_edge_counts.get(s, 0) + 1
            if t in output_indices:
                output_edge_counts[t] = output_edge_counts.get(t, 0) + 1
        
        # Check if constraints are violated
        for inp_idx in input_indices:
            if input_edge_counts.get(inp_idx, 0) != 1:
                edge_constraint_violations += 1
        for out_idx in output_indices:
            if output_edge_counts.get(out_idx, 0) != 1:
                edge_constraint_violations += 1
        
        # Check spider-webs
        webs = detect_spider_webs(colors, source, target)
        
        if has_gflow_simple:
            flow_count_simple += 1
        if has_gflow_pyzx:
            flow_count_pyzx += 1
        total_webs += webs['total_webs']
        total_nodes += len(colors)
        total_edges += len(source)
        total_inputs += len(input_indices)
        total_outputs += len(output_indices)
        
        print(f"Example {i+1}: {len(colors)} nodes, {len(source)} edges, "
              f"inputs={len(input_indices)}, outputs={len(output_indices)}, "
              f"gFlow(simple)={has_gflow_simple}, gFlow(PyZX)={has_gflow_pyzx}, "
              f"webs={webs['total_webs']}")
    
    print("\n" + "=" * 60)
    print("SUMMARY (10 examples)")
    print("=" * 60)
    print(f"Diagrams with flow (simplified): {flow_count_simple}/10 ({flow_count_simple/10*100:.1f}%)")
    print(f"Diagrams with flow (PyZX): {flow_count_pyzx}/10 ({flow_count_pyzx/10*100:.1f}%)")
    print(f"Average nodes: {total_nodes/10:.1f}")
    print(f"Average edges: {total_edges/10:.1f}")
    print(f"Average inputs: {total_inputs/10:.1f}")
    print(f"Average outputs: {total_outputs/10:.1f}")
    print(f"Average spider-webs: {total_webs/10:.2f}")
    print(f"Edge constraint violations: {edge_constraint_violations}")
    if edge_constraint_violations > 0:
        print(f"  ⚠ WARNING: {edge_constraint_violations} inputs/outputs don't have exactly one edge!")
    else:
        print(f"  ✓ All inputs/outputs have exactly one edge in all examples")
    print("=" * 60)


if __name__ == "__main__":
    main()

