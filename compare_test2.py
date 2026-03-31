import json
import sys

def run_boundary_test():
    """
    Analyzing the efficiency boundaries of Delta-CAS.
    Best Case: Small change in large state.
    Worst Case: Massive bulk update.
    """
    # Initialize a 100KB state (roughly 500 tasks)
    base_state = {f"t_{i}": "data_" * 20 for i in range(500)}
    full_size = len(json.dumps(base_state).encode('utf-8'))

    # --- SCENARIO A: THE "SWEET SPOT" (99% Savings) ---
    # Changing 1 field (Best Case)
    delta_small = {"t_0": "done"}
    size_small = len(json.dumps(delta_small).encode('utf-8'))

    # --- SCENARIO B: THE "BREAK-EVEN" POINT (Worst Case) ---
    # Changing 50% of the tasks simultaneously (Massive Bulk Update)
    # This simulates a 'Rebase' or a massive state migration.
    delta_large = {f"t_{i}": "updated_data_" * 20 for i in range(250)}
    size_large = len(json.dumps(delta_large).encode('utf-8'))

    print(f"Base State Size: {full_size / 1024:.2f} KB")
    print("-" * 40)
    print(f"Best Case (1 field): {size_small} bytes ({(1-size_small/full_size)*100:.2f}% saved)")
    print(f"Worst Case (50% change): {size_large / 1024:.2f} KB ({(1-size_large/full_size)*100:.2f}% saved)")
    
    if size_large > full_size * 0.8:
        print("\n[PRO TIP] When delta exceeds 80%, Delta-CAS triggers an automatic compaction/snapshot.")

run_boundary_test()
