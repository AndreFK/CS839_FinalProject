#!/usr/bin/env python3
"""
Test script to verify PyZX gflow implementation.
Compares the new PyZX-based gflow check with the simplified check.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zxreinforce.Resetters import Resetter_ZERO_PI_PIHALF_ARB_hada, Resetter_FlowPatterns
from zxreinforce.flow_checking import has_flow, check_gflow_pyzx, check_gflow
from zxreinforce.own_constants import INPUT, OUTPUT

def test_gflow_methods(resetter_class, resetter_name, n_tests=10):
    """Test both gflow checking methods."""
    print(f"\n{'='*70}")
    print(f"Testing {resetter_name}")
    print(f"{'='*70}")
    
    # Create resetter
    rng = np.random.default_rng(42)
    
    if resetter_name == "Resetter_ZERO_PI_PIHALF_ARB_hada":
        resetter = resetter_class(
            n_in_min=1,
            n_in_max=3,
            min_spiders=5,
            max_spiders=15,
            pi_fac=0.5,
            pi_half_fac=0.5,
            arb_fac=0.5,
            p_hada=0.1,
            min_mean_neighbours=2,
            max_mean_neighbours=4,
            rng=rng
        )
    else:  # Resetter_FlowPatterns
        resetter = resetter_class(
            n_in_min=1,
            n_in_max=3,
            n_out_min=1,
            n_out_max=3,
            n_layers_min=2,
            n_layers_max=5,
            nodes_per_layer_min=2,
            nodes_per_layer_max=5,
            pi_fac=0.5,
            pi_half_fac=0.5,
            arb_fac=0.5,
            p_hada=0.1,
            p_edge=0.3,
            rng=rng
        )
    
    pyzx_has_flow = 0
    simplified_has_flow = 0
    both_agree = 0
    both_have_flow = 0
    both_no_flow = 0
    disagreements = []
    
    for i in range(n_tests):
        try:
            colors, angles, selected_node, source, target, selected_edges = resetter.reset()
            
            # Check with PyZX method
            pyzx_result = check_gflow_pyzx(colors, source, target, INPUT, OUTPUT, angles=angles)
            pyzx_flow = pyzx_result is not None
            
            # Check with simplified method
            simplified_flow, _ = check_gflow(colors, source, target, INPUT, OUTPUT)
            
            # Check with has_flow (which uses PyZX first, then simplified)
            has_flow_result = has_flow(colors, source, target, INPUT, OUTPUT, angles=angles)
            
            if pyzx_flow:
                pyzx_has_flow += 1
            if simplified_flow:
                simplified_has_flow += 1
            if pyzx_flow == simplified_flow:
                both_agree += 1
                if pyzx_flow:
                    both_have_flow += 1
                else:
                    both_no_flow += 1
            else:
                disagreements.append({
                    'test': i+1,
                    'pyzx': pyzx_flow,
                    'simplified': simplified_flow,
                    'has_flow': has_flow_result,
                    'nodes': len(colors),
                    'edges': len(source)
                })
            
            status_pyzx = "✓" if pyzx_flow else "✗"
            status_simplified = "✓" if simplified_flow else "✗"
            status_has_flow = "✓" if has_flow_result else "✗"
            
            print(f"  Test {i+1}: PyZX={status_pyzx} Simplified={status_simplified} has_flow()={status_has_flow} "
                  f"(nodes: {len(colors)}, edges: {len(source)})")
            
            if pyzx_result is not None:
                layers, gflow = pyzx_result
                max_layer = max(layers.values()) if layers else 0
                print(f"    PyZX layers: {max_layer}, gflow corrections: {len(gflow)}")
            
        except Exception as e:
            print(f"  Test {i+1}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n  Results:")
    print(f"    PyZX method: {pyzx_has_flow}/{n_tests} diagrams have flow ({100*pyzx_has_flow/n_tests:.1f}%)")
    print(f"    Simplified method: {simplified_has_flow}/{n_tests} diagrams have flow ({100*simplified_has_flow/n_tests:.1f}%)")
    print(f"    Agreement: {both_agree}/{n_tests} ({100*both_agree/n_tests:.1f}%)")
    print(f"    Both have flow: {both_have_flow}/{n_tests}")
    print(f"    Both no flow: {both_no_flow}/{n_tests}")
    
    if disagreements:
        print(f"\n  Disagreements ({len(disagreements)}):")
        for d in disagreements:
            print(f"    Test {d['test']}: PyZX={d['pyzx']}, Simplified={d['simplified']}, "
                  f"has_flow()={d['has_flow']} (nodes: {d['nodes']}, edges: {d['edges']})")
    
    return {
        'pyzx_has_flow': pyzx_has_flow,
        'simplified_has_flow': simplified_has_flow,
        'both_agree': both_agree,
        'disagreements': len(disagreements)
    }


def test_simple_cases():
    """Test some simple known cases."""
    print(f"\n{'='*70}")
    print("Testing Simple Cases")
    print(f"{'='*70}")
    
    from zxreinforce.own_constants import GREEN, RED, ZERO, NO_ANGLE
    
    # Case 1: Simple chain: Input -> Spider -> Output
    print("\n  Case 1: Simple chain (Input -> Spider -> Output)")
    colors = np.array([INPUT, GREEN, OUTPUT])
    angles = np.array([NO_ANGLE, ZERO, NO_ANGLE])
    source = np.array([0, 1])
    target = np.array([1, 2])
    
    result = check_gflow_pyzx(colors, source, target, INPUT, OUTPUT, angles=angles)
    if result:
        print(f"    ✓ Has gflow: layers={result[0]}")
    else:
        print(f"    ✗ No gflow")
    
    # Case 2: Two inputs, one output
    print("\n  Case 2: Two inputs -> Spider -> Output")
    colors = np.array([INPUT, INPUT, GREEN, OUTPUT])
    angles = np.array([NO_ANGLE, NO_ANGLE, ZERO, NO_ANGLE])
    source = np.array([0, 1, 2])
    target = np.array([2, 2, 3])
    
    result = check_gflow_pyzx(colors, source, target, INPUT, OUTPUT, angles=angles)
    if result:
        print(f"    ✓ Has gflow: layers={result[0]}")
    else:
        print(f"    ✗ No gflow")
    
    # Case 3: Cycle (should not have flow)
    print("\n  Case 3: Cycle (Input -> Spider1 <-> Spider2 -> Output)")
    colors = np.array([INPUT, GREEN, GREEN, OUTPUT])
    angles = np.array([NO_ANGLE, ZERO, ZERO, NO_ANGLE])
    source = np.array([0, 1, 2, 2])
    target = np.array([1, 2, 1, 3])
    
    result = check_gflow_pyzx(colors, source, target, INPUT, OUTPUT, angles=angles)
    if result:
        print(f"    ✓ Has gflow: layers={result[0]}")
    else:
        print(f"    ✗ No gflow (expected)")


if __name__ == "__main__":
    print("Testing PyZX gflow implementation...")
    print("This compares the new PyZX-based gflow check with the simplified check.\n")
    
    # Test simple cases first
    test_simple_cases()
    
    # Test both resetters
    test1 = test_gflow_methods(
        Resetter_ZERO_PI_PIHALF_ARB_hada, 
        "Resetter_ZERO_PI_PIHALF_ARB_hada",
        n_tests=20
    )
    
    test2 = test_gflow_methods(
        Resetter_FlowPatterns,
        "Resetter_FlowPatterns",
        n_tests=20
    )
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Resetter_ZERO_PI_PIHALF_ARB_hada:")
    print(f"  PyZX flow: {test1['pyzx_has_flow']}/20, Simplified: {test1['simplified_has_flow']}/20")
    print(f"  Agreement: {test1['both_agree']}/20, Disagreements: {test1['disagreements']}")
    print(f"\nResetter_FlowPatterns:")
    print(f"  PyZX flow: {test2['pyzx_has_flow']}/20, Simplified: {test2['simplified_has_flow']}/20")
    print(f"  Agreement: {test2['both_agree']}/20, Disagreements: {test2['disagreements']}")
    
    if test1['disagreements'] == 0 and test2['disagreements'] == 0:
        print("\n✓ All methods agree!")
    else:
        print(f"\n⚠ Found {test1['disagreements'] + test2['disagreements']} disagreements between methods.")







