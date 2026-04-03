import json
import time

# ---------------------------------------------------------
# SCENARIO: A complex Agent State (e.g., Task Management)
# ---------------------------------------------------------
# Imagine your agent maintains a large state tree with 100 tasks.
# Total size is roughly 25KB.
complex_state = {
    f"task_{i}": {
        "id": i, 
        "status": "pending", 
        "priority": "high",
        "description": "An extremely detailed task description that consumes tokens..." * 10
    }
    for i in range(100)
}

def compare_transmission_volume():
    """
    Demonstrates the 'Save your Tokens' value proposition.
    Compares full-state snapshot sync vs. Delta-CAS incremental update.
    """
    
    # 1. LEGACY APPROACH (Full Snapshot)
    # Every time the agent updates ONE field, it sends the WHOLE object.
    full_payload = json.dumps(complex_state)
    legacy_size = len(full_payload.encode('utf-8'))

    # 2. DELTA-CAS APPROACH (Dot-notation Incremental)
    # Only send the specific path that changed.
    # Ref: test_dot_notation_three_levels in test_core.py
    delta_payload = {"task_42.status": "completed"}
    delta_cas_size = len(json.dumps(delta_payload).encode('utf-8'))

    # ---------------------------------------------------------
    # ANALYTICS
    # ---------------------------------------------------------
    savings = (1 - delta_cas_size / legacy_size) * 100
    
    print("="*50)
    print(" TRANSMISSION VOLUME COMPARISON (Single Update)")
    print("="*50)
    print(f"Legacy Sync (Full JSON):  {legacy_size:,} bytes")
    print(f"Delta-CAS (Incremental):   {delta_cas_size:,} bytes")
    print(f"Token Savings:             {savings:.2f}%")
    print("-"*50)
    print("CONCLUSION: In a multi-step workflow, Delta-CAS prevents")
    print("context window bloat and slashes LLM costs.")
    print("="*50)

if __name__ == "__main__":
    compare_transmission_volume()
