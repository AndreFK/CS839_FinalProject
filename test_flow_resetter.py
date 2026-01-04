#!/usr/bin/env python3
"""
Script to visualize and test the Resetter_FlowPatterns output.

This script generates a ZX diagram using Resetter_FlowPatterns and displays:
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

from zxreinforce.Resetters import Resetter_FlowPatterns
from zxreinforce.ZX_env import ZXCalculus
from zxreinforce.plot_utils import plot_observation
from zxreinforce.flow_checking import check_gflow, has_flow
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
    
    # Check flow
    has_gflow, flow_depth = check_gflow(colors, source, target, INPUT, OUTPUT)
    
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
    print(f"  - Has gFlow: {has_gflow}")
    print(f"  - Flow depth: {flow_depth}")
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
    
    # Create resetter with reasonable parameters
    resetter = Resetter_FlowPatterns(
        n_in_min=2,
        n_in_max=4,
        n_out_min=2,
        n_out_max=4,
        n_layers_min=2,
        n_layers_max=4,
        nodes_per_layer_min=3,
        nodes_per_layer_max=6,
        p_edge=0.5,
        pi_fac=0.5,
        pi_half_fac=0.5,
        arb_fac=0.5,
        p_hada=0.1,
        rng=rng
    )
    
    print("Generating diagram with Resetter_FlowPatterns...")
    print(f"Parameters:")
    print(f"  - Inputs: {resetter.n_in_min}-{resetter.n_in_max}")
    print(f"  - Outputs: {resetter.n_out_min}-{resetter.n_out_max}")
    print(f"  - Layers: {resetter.n_layers_min}-{resetter.n_layers_max}")
    print(f"  - Nodes per layer: {resetter.nodes_per_layer_min}-{resetter.nodes_per_layer_max}")
    print(f"  - Edge probability: {resetter.p_edge}")
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
    plt.figure(figsize=(14, 10))
    plot_observation(observation)
    plt.title("ZX Diagram Generated by Resetter_FlowPatterns", fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save plot to file
    output_file = "flow_diagram_example.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Diagram saved to: {output_file}")
    plt.close()
    
    # Generate a few more examples
    print("\n" + "=" * 60)
    print("Generating 5 more examples for statistics...")
    print("=" * 60)
    
    flow_count = 0
    total_webs = 0
    total_nodes = 0
    total_edges = 0
    
    for i in range(5):
        colors, angles, selected_node, source, target, selected_edges = resetter.reset()
        has_gflow, _ = check_gflow(colors, source, target, INPUT, OUTPUT)
        webs = detect_spider_webs(colors, source, target)
        
        if has_gflow:
            flow_count += 1
        total_webs += webs['total_webs']
        total_nodes += len(colors)
        total_edges += len(source)
        
        print(f"Example {i+1}: {len(colors)} nodes, {len(source)} edges, "
              f"gFlow={has_gflow}, webs={webs['total_webs']}")
    
    print("\n" + "=" * 60)
    print("SUMMARY (5 examples)")
    print("=" * 60)
    print(f"Diagrams with flow: {flow_count}/5 ({flow_count/5*100:.1f}%)")
    print(f"Average nodes: {total_nodes/5:.1f}")
    print(f"Average edges: {total_edges/5:.1f}")
    print(f"Average spider-webs: {total_webs/5:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

