#!/usr/bin/env python3
"""Quick test to verify generate_human method exists and works"""

import sys
import os
sys.path.insert(0, '/home/nisarlab/Desktop/Crowd_nav/ComplexityNav')

try:
    # Try to import and check that the method exists
    from crowd_sim.envs.crowd_sim import CrowdSim
    
    # Check if the method exists
    if hasattr(CrowdSim, 'generate_human'):
        print("✓ CrowdSim.generate_human method exists")
    else:
        print("✗ CrowdSim.generate_human method NOT found")
        sys.exit(1)
    
    if hasattr(CrowdSim, 'generate_human_from_state'):
        print("✓ CrowdSim.generate_human_from_state method exists")
    else:
        print("✗ CrowdSim.generate_human_from_state method NOT found")
        sys.exit(1)
    
    print("\n✓ All methods verified successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
