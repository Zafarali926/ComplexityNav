# Power Law & PLEdestrians Policy Integration

Documentation for adding **Power Law** and **PLEdestrians** as human navigation policies in ComplexityNav, alongside existing **ORCA** and **SFM** policies.

**Source references (UMANS C++ plugins, not used directly):**
- `crowd_nav/New agents/PowerLaw.cpp` / `PowerLaw.h` — Karamouzas et al. (2014)
- `crowd_nav/New agents/PLEdestrians.cpp` / `PLEdestrians.h` — Guy et al. (2010)

**Python implementations (used at runtime):**
- `crowd_sim/envs/policy/powerlaw.py`
- `crowd_sim/envs/policy/pledestrians.py`

---

## 1. ORCA / SFM vs Power Law / PLEdestrians

| | **ORCA** | **SFM (Social Force)** | **Power Law** | **PLEdestrians** |
|---|----------|------------------------|---------------|------------------|
| **Paper** | RVO / reciprocal velocity obstacles | Helbing social force model | Karamouzas et al. 2014 | Guy et al. 2010 |
| **Type** | Geometric velocity obstacle | Force-based | Force-based (TTC power law) | Cost-based velocity selection |
| **Output** | Collision-free velocity from RVO2 | Velocity from force integration | Force sum → velocity update | Best sampled velocity (min cost) |
| **Goal term** | Preferred velocity toward goal | Goal attraction force | Separate goal force (added in Python) | Built into cost function |
| **Avoidance** | Hard ORCA constraints | Exponential repulsion | Soft TTC-based force | Reject unsafe velocities (`ttc < t_min`) |
| **Implementation** | `orca.py` + Python-RVO2 (C++) | `socialforce.py` + PyTorch `socialforce` pkg | Python port of UMANS math | Python port + velocity sampling |
| **Obstacles** | ORCA borders supported | SFM borders (partial) | Not in original UMANS force | Neighbor discs only |

All four use the **same human body radius** from config (`humans.radius = 0.3` m). Power Law and PLEdestrians also use `safety_space = 0.01` m per agent in collision math (same buffer idea as ORCA’s `radius + 0.01`).

---

## 2. What “concrete Python API” meant

Before implementation, the proposal was to define new policy classes matching the existing pattern:

```python
class PowerLaw(Policy):
    def predict(self, state, border=None) -> ActionXY:
        # decentralized: state is JointState (robot_state + human_states)

class CentralizedPowerLaw(PowerLaw):
    def predict(self, state, border=None) -> list[ActionXY]:
        # centralized: state is list[FullState] — one per agent
```

- **Input:** `FullState` fields used: `px, py, vx, vy, radius, gx, gy, v_pref`
- **Output:** `ActionXY(vx, vy)` — holonomic velocity for one `time_step` (0.25 s)

This mirrors `orca.py` / `socialforce.py`. The C++ UMANS files cannot plug in directly; the math was **reimplemented in Python**.

---

## 3. Files created

### `crowd_sim/envs/policy/powerlaw.py`

| Class | Role |
|-------|------|
| `PowerLaw` | Single-agent policy (`predict` on `JointState`) |
| `CentralizedPowerLaw` | Multi-human centralized planning (`predict` on `list[FullState]`) |

**Algorithm:**
1. Goal force toward `(gx, gy)` at `v_pref` (UMANS `GoalReachingForce` analogue).
2. Per-neighbor Power Law interaction force (ported from `PowerLaw.cpp`).
3. Integrate: `v_new = v + force * dt`, clip to `v_pref`.
4. Force magnitude capped at 10.0 for stability.

**Default parameters:** `k=1.5`, `tau0=3.0`, `tau_g=0.5`, `safety_space=0.01`, `neighbor_dist=10.0`.

### `crowd_sim/envs/policy/pledestrians.py`

| Class | Role |
|-------|------|
| `PLEdestrians` | Single-agent policy |
| `CentralizedPLEdestrians` | Multi-human centralized planning |

**Algorithm:**
1. Sample candidate velocities (16 directions × 5 speeds + goal direction + stop).
2. Score each with PLEdestrians cost (ported from `PLEdestrians.cpp`).
3. Pick minimum cost; if all unsafe, fallback to goal-directed velocity.

**Default parameters:** `w_a=2.23`, `w_b=1.26`, `t_min=0.5`, `t_max=3.0`, `safety_space=0.01`.

---

## 4. Files modified

### `crowd_sim/envs/policy/policy_factory.py`

**Added registrations (ORCA/SFM unchanged):**

```python
policy_factory['powerlaw'] = PowerLaw
policy_factory['pledestrians'] = PLEdestrians
policy_factory['centralized_powerlaw'] = CentralizedPowerLaw
policy_factory['centralized_pledestrians'] = CentralizedPLEdestrians
```

### `crowd_sim/envs/policy/multi_policy.py`

`CentralizedMultiPolicy.predict()` now:
1. Runs `CentralizedPowerLaw` and `CentralizedPLEdestrians` once per step (like ORCA/SFM).
2. Dispatches per human via `isinstance`:

```python
elif isinstance(policies[i], PowerLaw):
    actions.append(powerlaw_actions[i])
elif isinstance(policies[i], PLEdestrians):
    actions.append(pledestrians_actions[i])
```

ORCA, SFM, and Linear branches are unchanged.

### `crowd_sim/envs/crowd_sim.py`

**`set_num_policies()`** — added keys alongside existing ones (did **not** remove ORCA/SFM):

```python
self.num_policies = {
    'orca': num_orca,
    'socialforce': num_sf,
    'powerlaw': num_powerlaw,        # NEW
    'pledestrians': num_pledestrians,  # NEW
    'linear': num_linear,
    'static': num_static,
}
```

**`render()`** — added `_human_policy_color()` so Power Law (orange `#e66101`) and PLEdestrians (purple `#9970ab`) are visible in animations.

### `crowd_nav/configs/icra_benchmark/config.py`

**`BaseEnvConfig` defaults:**

```python
humans.num_powerlaw = [0]
humans.num_pledestrians = [0]
```

**`BaseExperimentsConfig`** — added `exp.num_powerlaw` and `exp.num_pledestrians` arrays for benchmark rows `e=4` (policy mixture) and `e=5` (density sweep). Rows `e=0–3` keep zeros for PL/PLD.

### `crowd_sim/envs/utils/utils.py`

`generate_scenarios_fixed()` and `random_sequence()` extended to include `powerlaw` and `pledestrians` policy keys for pre-generated benchmark scenarios.

### `crowd_nav/human_mix_presets.py`

**New mix presets:**

| Preset | Composition |
|--------|-------------|
| `PL_only` | 15 Power Law |
| `PLD_only` | 15 PLEdestrians |
| `PL_mix1` | 8 Power Law + 7 PLEdestrians |
| `PL_mix2` | 5 PL + 5 PLD + 2 CV + 3 static |
| `PL_mix3` | 4 PL + 4 PLD + 4 CV + 3 static |

Existing `sfm_only`, `orca_only`, `mix1–3` unchanged (with `powerlaw: 0`, `pledestrians: 0`).

Also added:
- `BENCHMARK_MIX_CELLS` — maps `(exp_e=4, exp_se)` → PL presets
- `BENCHMARK_PL_DENSITY_CELLS` — maps `(exp_e=5, exp_se)` → PL/PLD density counts

### `crowd_nav/test.py`

- `apply_human_mix()` — sets `num_powerlaw`, `num_pledestrians`
- `use_config_human_mix()` — includes new CLI flags
- `apply_exp_cell()` — reads `ec.exp.num_powerlaw` / `num_pledestrians` from benchmark grid
- **New CLI flags:** `--num_powerlaw`, `--num_pledestrians`
- **`--human_mix`** help text includes `PL_only`, `PLD_only`

### `crowd_nav/utils/test_human_mixes.py`

- Imports `PowerLaw`, `PLEdestrians` for policy labels (`PL`, `PLD`)
- `apply_mix()` sets `num_powerlaw` / `num_pledestrians`
- `PL_only` / `PLD_only` use **`circle_crossing`** spawn via `CIRCLE_CROSSING_MIXES`

### `crowd_nav/policy/model_predictive_rl.py`

**New robot baselines** (alongside `orca`, `sfm`, `cv`):

```python
elif baseline == 'powerlaw':
    pl = PowerLaw()
    action = pl.predict(state, border=border)
elif baseline == 'pledestrians':
    pld = PLEdestrians()
    action = pld.predict(state, border=border)
```

### Visualization utilities

| File | Change |
|------|--------|
| `utils/visualize_benchmark_colab.py` | `powerlaw`/`pledestrians` columns in dataframe; density x-label updated; `e=4`/`e=5` benchmark rows |
| `utils/visualize_rgl_results.py` | Density axis includes PL/PLD counts |
| `utils/visualize_nav_models.py` | Docstring lists `powerlaw_results.json` / `pledestrians_results.json` (optional plot entries) |

---

## 5. What was NOT removed

- ORCA (`orca`, `centralized_orca`)
- SFM (`socialforce`, `centralized_socialforce`)
- Linear / static policies
- All original human mix presets and benchmark row `e=3`

---

## 6. Usage examples

### Human-only visualization (no robot)

```bash
cd crowd_nav

python utils/test_human_mixes.py --human_mix PL_only --visualize
python utils/test_human_mixes.py --human_mix PLD_only --visualize
```

`PL_only` / `PLD_only` use **circle_crossing** in `test_human_mixes.py`.

### Test with robot (RGL or other trained policy)

```bash
python test.py \
  --model_dir data/rgl_output \
  --config configs/icra_benchmark/rgl.py \
  --phase test -v \
  --human_mix PLD_only \
  --test_scenario circle_crossing
```

### Custom human counts

```bash
python test.py \
  --model_dir data/rgl_output \
  --config configs/icra_benchmark/rgl.py \
  --num_pledestrians 15 \
  --num_orca 0 --num_sf 0 --num_linear 0 --num_static 0 \
  --test_scenario circle_crossing -v
```

### Benchmark grid (pre-generated scenarios)

| Goal | Command |
|------|---------|
| ORCA/SFM policy mix (e.g. mix2) | `--exp_e 3 --exp_se 3` |
| PL/PLD policy mix (e.g. PL_mix2) | `--exp_e 4 --exp_se 3` |
| PL/PLD density (parallel to e=0) | `--exp_e 5 --exp_se 3` |

```bash
python test.py --model_dir data/rgl_output --config configs/icra_benchmark/rgl.py \
  --phase test -v --exp_e 4 --exp_se 1   # PLD_only equivalent
```

---

## 7. Architecture flow

```
EnvConfig / CLI / human_mix_presets
        ↓
crowd_sim.set_num_policies()  →  { powerlaw: N, pledestrians: M, orca: ..., sf: ... }
        ↓
generate_human(policy='powerlaw' | 'pledestrians')
        ↓
policy_factory['powerlaw']() | policy_factory['pledestrians']()
        ↓
CrowdSim.step() → CentralizedMultiPolicy.predict()
        ↓
CentralizedPowerLaw / CentralizedPLEdestrians  →  ActionXY per human
```

---

## 8. Policy factory keys (quick reference)

| Key | Class |
|-----|-------|
| `orca` | `ORCA` |
| `socialforce` | `SocialForce` |
| `powerlaw` | `PowerLaw` |
| `pledestrians` | `PLEdestrians` |
| `linear` | `Linear` |
| `centralized_multipolicy` | `CentralizedMultiPolicy` |

Human `policy` string in `generate_human()` must match factory keys (`'socialforce'` not `'sfm'`).

---

## 9. Limitations (Python ports)

1. **Not exact UMANS binaries** — math reimplemented from `crowd_nav/New agents/*.cpp`.
2. **Power Law** — no obstacle force (same gap as UMANS header); goal force added in Python.
3. **PLEdestrians** — uses velocity sampling, not closed-form optimum from the paper.
4. **Crowded scenes** — PLEdestrians may pick goal fallback when all samples are unsafe (`ttc < t_min`).

---

## 10. Related later changes (separate from initial integration)

| Doc / change | Description |
|--------------|-------------|
| `docs/RGL_PLEDESTRIANS_TRAINING.md` | RGL training with 15× PLEdestrians + `circle_crossing` in `rgl.py`; `best_val.pth` every 1000 eps |
| Render fix | Removed empty second matplotlib figure; human colors for PL/PLD |
| `rgl.py` `EnvConfig` | Training defaults for PLEdestrians + circle crossing |

---

## 11. File change checklist

| File | Created / Modified |
|------|-------------------|
| `crowd_sim/envs/policy/powerlaw.py` | **Created** |
| `crowd_sim/envs/policy/pledestrians.py` | **Created** |
| `crowd_sim/envs/policy/policy_factory.py` | Modified |
| `crowd_sim/envs/policy/multi_policy.py` | Modified |
| `crowd_sim/envs/crowd_sim.py` | Modified |
| `crowd_nav/configs/icra_benchmark/config.py` | Modified |
| `crowd_nav/human_mix_presets.py` | Modified |
| `crowd_nav/test.py` | Modified |
| `crowd_nav/utils/test_human_mixes.py` | Modified |
| `crowd_nav/utils/visualize_benchmark_colab.py` | Modified |
| `crowd_nav/utils/visualize_rgl_results.py` | Modified |
| `crowd_nav/utils/visualize_nav_models.py` | Modified |
| `crowd_nav/policy/model_predictive_rl.py` | Modified |
| `crowd_sim/envs/utils/utils.py` | Modified (scenario generation) |
