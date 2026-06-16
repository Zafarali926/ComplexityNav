# Fix Summary: Missing `generate_human()` Method

## Issue
Training failed with:
```
AttributeError: 'CrowdSim' object has no attribute 'generate_human'
```

**Location**: [crowd_sim/envs/crowd_sim.py](crowd_sim/envs/crowd_sim.py) line 272 and 303 in `reset()` method

## Root Cause
The `reset()` method tried to call `self.generate_human()` but the method was not defined. Only `generate_human_from_state()` existed, which requires a pre-generated state tuple.

## Solution Applied
Added a new `generate_human(human=None, policy=None)` method to the `CrowdSim` class that:

1. **Creates a Human instance** with appropriate policy based on mode:
   - Multi-policy mode: Uses specified policy from factory (e.g., `policy_factory['orca']`)
   - Single-policy mode: Uses default policy from config
   - Static humans: Special handling with v_pref=1e-4

2. **Generates random valid state** by:
   - Calling `utils.generate_human_state()` to sample position/velocity/goal
   - Samples respect scenario type (passing, crossing, circle_crossing, random)
   - Validates no collision with existing agents
   - Current_scenario determines distribution

3. **Sets state on human** with `human.set(*human_state)`

## Files Modified
- **[crowd_sim/envs/crowd_sim.py](crowd_sim/envs/crowd_sim.py#L202-L244)**: Added `generate_human()` method (lines 202-228)
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)**: Added section 6 documenting both human generation methods

## Key Patterns
Both human generation methods now properly documented:
- **Random generation** (`generate_human`): Scenario-based sampling, variable outcomes
- **State-based generation** (`generate_human_from_state`): Deterministic, used for pre-generated scenarios

## Testing
- Syntax validation: ✓ No syntax errors
- Method existence: ✓ Both methods now present and callable
- Call signatures match usage patterns: ✓
  - `self.generate_human()` - creates with random state
  - `self.generate_human(human=None, policy=policy_name)` - multi-policy with random state
  - `self.generate_human_from_state(policy, state_tuple)` - deterministic with provided state
