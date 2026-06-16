# AI Coding Agent Instructions for ComplexityNav

## Project Overview
ComplexityNav studies complexity factors in social robot navigation using PyTorch/OpenAI Gym. The codebase has two main components: **crowd_sim/** (physics simulation) and **crowd_nav/** (policy training/testing). It builds on RelationalGraphLearning but adds static obstacles, multi-policy human behaviors, and model predictive control methods (SGAN, MPC, MPPI).

## Critical Architecture Patterns

### 1. Config-Driven Architecture
**Pattern**: All training/testing uses hierarchical Python config classes (not YAML).
- **Inheritance chain**: `BaseXConfig` → `XConfig` (e.g., `BaseEnvConfig` → `EnvConfig`)
- **Location**: Configs in `crowd_nav/configs/icra_benchmark/` with policy-specific overrides (rgl.py, sarl.py, etc.)
- **Key classes**: `ExperimentsConfig`, `EnvConfig`, `PolicyConfig`, `TrainConfig`
- **Important**: Config is **copied to output_dir** during training (`train.py` line ~43) so reproducibility is maintained
- **Example**: [crowd_nav/configs/icra_benchmark/config.py](crowd_nav/configs/icra_benchmark/config.py#L1-L50) shows nested Config objects (exp, sim, reward, humans, MPC)

### 2. Gym Environment with Agents
**Pattern**: `CrowdSim` (gym.Env) manages n+1 agents: 1 Robot + n Humans
- **Reset/Step**: `env.reset(phase, scenario, goals)` returns observation; `env.step(action)` returns (obs, reward, done, info)
- **Key attributes**: `robot` (trainable), `humans` (list, fixed policies), `global_time`, `time_step=0.25`
- **Collision detection**: Built-in during `step()`, triggers reward penalty and episode termination
- **Observation format**: `robot.get_observation()` returns joint state of robot + all visible humans (coordinates-based)
- **File**: [crowd_sim/envs/crowd_sim.py](crowd_sim/envs/crowd_sim.py#L1-L50)

### 3. Policy Factory Pattern
**Pattern**: Policies are registered in a factory dict; instantiated by name
- **Human policies**: ORCA, SocialForce, Linear (in `crowd_sim/envs/policy/`)
- **Robot policies**: CADRL, LSTM-RL, SARL, GCN, ModelPredictiveRL, vecMPC, vecMPPI
- **File**: [crowd_nav/policy/policy_factory.py](crowd_nav/policy/policy_factory.py) registers robot policies
- **Usage**: `policy_factory['gcn']` creates GCN policy instance
- **Base class**: All robot policies inherit behavior from `crowd_sim.envs.policy.policy` interface with `act()` method

### 4. Training Loop: Explorer + Trainer
**Pattern**: Two-stage training separates episode exploration from model updates
- **Explorer**: Runs k episodes, collects trajectories (states/actions/rewards), returns statistics
  - File: [crowd_nav/utils/explorer.py](crowd_nav/utils/explorer.py#L1-L80)
  - Key method: `run_k_episodes(k, phase, update_memory)` 
  - Computes: success/collision/timeout counts, min_distances, path irregularity
- **Trainer** (MPRLTrainer): Updates value_estimator + state_predictor
  - File: [crowd_nav/utils/trainer.py](crowd_nav/utils/trainer.py#L1-L80)
  - Updates frequency: state_predictor updated every `human_num` steps (controlled by `reduce_sp_update_frequency`)
- **Memory**: ReplayMemory stores (robot_state, human_states, value, actions, rewards, next_human_states)
- **Main loop** in [train.py](crowd_nav/train.py#L1-L80): alternates `explorer.run_k_episodes()` → `trainer.optimize_epoch()`

### 5. State Representation
**Pattern**: Multiple state types serve different purposes
- **ObservableState**: (px, py, vx, vy, radius) - what other agents see
- **FullState**: ObservableState + (gx, gy, v_pref, theta) - internal to agent
- **JointState**: (robot_full_state, [all_human_observable_states])
- **DualState**: (robot_full_state, one_human_observable_state) - for CADRL
- **Conversion**: `robot.get_full_state()` (FullState) vs `robot.get_observation()` (JointState)
- **File**: [crowd_sim/envs/utils/state.py](crowd_sim/envs/utils/state.py)

### 6. Human Agent Generation
**Pattern**: Two methods for creating humans with different initialization approaches
- **`generate_human(human=None, policy=None)`**: Creates human with **random state** (position/velocity/goal sampled from scenario distributions)
  - Used in non-multi-policy reset: `self.humans.append(self.generate_human())`
  - Used in multi-policy reset: `self.humans.append(self.generate_human(human=None, policy=policy_name))`
  - Calls `utils.generate_human_state()` to sample valid scenario-specific position/goal pairs avoiding collisions
- **`generate_human_from_state(policy, state, human=None)`**: Creates human with **specified state** (position/velocity/goal provided)
  - Used when loading pre-generated scenarios: `human = self.generate_human_from_state(policy, state_tuple)`
  - Called during multi-policy reset with predefined scenario data from [utils/utils.py](crowd_sim/envs/utils/utils.py)
- **Key difference**: Random generation samples from scenario type (passing/crossing/random); state-based is deterministic
- **File**: [crowd_sim/envs/crowd_sim.py](crowd_sim/envs/crowd_sim.py#L202-L244)

## Developer Workflows

### Training a Policy
```bash
# From crowd_nav/ directory (see context terminal path)
python train.py --policy gcn --config crowd_nav/configs/icra_benchmark/rgl.py --output_dir data/output/gcn_test
# With GPU: add --gpu
# Resume training: add --resume
```
**Output**: Saves to `data/output/` with structure: `config.py`, `output.log`, `il_model.pth`, `rl_model_*.pth`

### Testing & Visualization
```bash
# Single episode with visualization
python test.py --policy gcn --model_dir data/output/gcn_test --phase test --visualize

# Batch testing (no viz, generates JSON results)
python test.py --policy gcn --model_dir data/output/gcn_test --phase test --test_size 100
```
**Output**: Appends results to `results.json` in codebase root

### Plotting Training Curves
```bash
python utils/plot.py data/output/gcn_test/output.log  # Plots from training log
```

## Project-Specific Patterns

### Multi-Policy Human Behaviors
**Innovation**: Unlike standard CrowdNav, this repo supports heterogeneous human policies in one scenario
- **Config field**: `humans.policy = 'multipolicy'` enables it
- **Mix setup**: `num_orca`, `num_sf`, `num_linear`, `num_static` (separate lists per scenario)
- **Random assignment**: During `env.reset()`, humans are randomly assigned policies based on counts
- **Use case**: Tests robot robustness across diverse pedestrian behaviors (not just single-policy crowds)
- **File**: [crowd_sim/envs/crowd_sim.py](crowd_sim/envs/crowd_sim.py#L80-L120) in `set_num_policies()` method

### MPC/MPPI Integration
**Files**: `crowd_nav/policy/model_predictive_rl.py`, `vecMPC/`, `vecMPC/` subdirs
- **SGAN trajectory predictor**: Loaded from YAML config path (see `MPC.path` in config)
- **Predictive horizon**: Configured in `exp.horizon` (typically 3-7 steps)
- **Update frequency**: MPC re-plans every step; MPPI is sampling-based alternative
- **Note**: Requires matching SGAN model weights; path hardcoded in config (consider env vars for portability)

### Pedestrian Group Detection
**New feature**: Computes groups of pedestrians for scenario complexity analysis
- **Methods**: DBScan, Coherent Filter (in `crowd_sim/envs/grouping/`)
- **Visualization**: Groups rendered as polygons (Convex Hull or Perimeter) in test plots
- **Files**: [crowd_sim/envs/grouping/README.md](crowd_sim/envs/grouping/README.md) documents grouping algorithms
- **Used for**: Complexity metrics in paper; not directly affecting policy training

### Complexity Factors
**Paper focus**: Environment complexity is varied via:
- **Scenario types**: passing, crossing, passing_crossing, circle_crossing, random
- **Density**: Varies human count (2-18 per scenario)
- **Obstacles**: Static obstacles added via `num_static` count
- **Directionality**: Humans may move in opposite/orthogonal directions (configured in scenario)
- **Heterogeneity**: Mix of ORCA/SocialForce/Linear policies (tests robustness)

## Important Implementation Details

### Reward Function
```python
reward = success_reward * done_success  # +1 if goal reached
       - collision_penalty * done_collision  # -0.25 if collided
       - discomfort_penalty_factor * (1 / (min_dist + 1e-5)) if min_dist < discomfort_dist  # Smooth personal space
```
**File**: [crowd_sim/envs/crowd_sim.py](crowd_sim/envs/crowd_sim.py#L200-L250) (compute_reward method)

### Action Space
- **Holonomic robots**: ActionXY(vx, vy) ∈ [-1, 1]²
- **Unicycle robots**: ActionRot(v, θ) with rotation constraint
- **Clip & normalize**: Actions clipped to agent velocity bounds during step()

### Logging & Debugging
- **Default**: INFO level to stdout + file
- **Debug**: Add `--debug` flag to train.py/test.py for DEBUG level with detailed logs
- **TensorBoard**: `SummaryWriter` logs training curves to `data/output/` (plots with `utils/plot.py`)

## Common Pitfalls

1. **Config mismatch**: Ensure policy name in config matches registered policy (typos cause cryptic errors)
2. **Phase switching**: Policies detect phase (train/val/test) via `policy.set_phase(phase)` to adjust behavior (e.g., exploration vs exploitation)
3. **Observation format**: Robot receives JointState (all humans) but CADRL expects DualState (one human at a time); SARL/GCN handle JointState natively
4. **Random seeds**: Set `env.random_seed = True` if reproducibility needed (affects human position init); also call `set_random_seeds()` in train.py
5. **State predictor freezing**: `freeze_state_predictor=True` in TrainConfig disables state_predictor updates (useful for pretrained models)

## File Navigation Reference

| Task | Primary Files |
|------|---|
| Add new policy | `crowd_nav/policy/my_policy.py`, register in `policy_factory.py` |
| Add new human policy | `crowd_sim/envs/policy/my_policy.py`, register in `crowd_sim/envs/policy/policy_factory.py` |
| Modify reward | `crowd_sim/envs/crowd_sim.py` compute_reward() |
| Add metric/statistic | `crowd_nav/utils/explorer.py` run_k_episodes() return dict |
| Change scenario generation | `crowd_sim/envs/utils/utils.py` (random_sequence, scenario generators) |
| Tune hyperparameters | `crowd_nav/configs/icra_benchmark/{rgl,sarl,mp_*.py}` |

## Key Dependencies
- PyTorch 1.13 (GPU compute for policies)
- OpenAI Gym (environment interface)
- Python-RVO2 (collision avoidance baseline)
- SocialForce library (human motion model)
- Pytorch MPPI (sampling-based control)
- Matplotlib (visualization)
