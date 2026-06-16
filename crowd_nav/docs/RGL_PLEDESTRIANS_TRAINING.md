# RGL Training with PLEdestrians + Circle Crossing

This document describes changes made to train **RGL (GCN policy)** with **PLEdestrians** human agents in a **circle_crossing** scenario, and to persist **`best_val.pth` every 1000 episodes**.

---

## Summary of changes

| File | Change |
|------|--------|
| `configs/icra_benchmark/rgl.py` | `EnvConfig` overrides: `circle_crossing`, 15× PLEdestrians only |
| `train.py` | Save / update `best_val.pth` on every `checkpoint_interval` (default 1000) after validation |

No changes were required to `pledestrians.py`, `policy_factory.py`, or `multi_policy.py` — PLEdestrians was already registered and wired.

---

## 1. `configs/icra_benchmark/rgl.py`

### Before
`EnvConfig` inherited defaults from `BaseEnvConfig`:
- `sim.test_scenario = 'passing_crossing'`
- Mixed humans: 4 ORCA + 4 SFM + 4 linear + 3 static

### After
`EnvConfig` now sets:

```python
self.sim.test_scenario = 'circle_crossing'
self.sim.train_val_scenario = 'circle_crossing'
self.humans.num_pledestrians = [15]
self.humans.num_powerlaw = [0]
self.humans.num_orca = [0]
self.humans.num_sf = [0]
self.humans.num_linear = [0]
self.humans.num_static = [0]
```

### Effect
- Humans spawn on a circle (radius 4 m) with goals on the opposite side.
- All 15 humans use the **PLEdestrians** policy via centralized multi-policy planning.
- Robot still trains with **GCN** (`PolicyConfig.name = 'gcn'`).
- Imitation learning still uses **ORCA** as the robot expert (`BaseTrainConfig.il_policy = 'orca'`).

---

## 2. `train.py` — `best_val.pth` checkpointing

### Before
- Every 1000 episodes: validation ran; `best_val_model` updated in memory only if val reward improved.
- `rl_model_9_<N>.pth` saved every 1000 episodes.
- **`best_val.pth` written once** at the end of all training.

### After
- Every `checkpoint_interval` episodes (default **1000**), after validation:
  1. If val cumulative reward **improves**, update in-memory `best_val_model`.
  2. If `best_val_model` exists, **write `best_val.pth`** to `output_dir` (create or overwrite).
- End-of-training save of `best_val.pth` is **kept** as a final safety write.

### Log message
```
Saved best_val.pth at episode 1000 (best val cumulative reward: X.XXXX)
```

### Unchanged
- `evaluation_interval = 1000`
- `checkpoint_interval = 1000`
- `rl_model_9_<N>.pth` still saved every 1000 episodes
- `il_model.pth` after imitation learning

---

## Training command

```bash
cd crowd_nav

python train.py \
  --config configs/icra_benchmark/rgl.py \
  --output_dir data/rgl_pledestrians_cc \
  --gpu \
  --gpu_id 0
```

Outputs in `data/rgl_pledestrians_cc/`:
- `config.py` — copy of training config
- `il_model.pth` — after imitation learning
- `best_val.pth` — updated at episodes 1000, 2000, … (best val reward so far)
- `rl_model_9_0.pth`, `rl_model_9_1.pth`, … — periodic full model snapshots
- `output.log`, TensorBoard events

---

## Testing command (popup visualization)

Use the **saved config** from the training run so env matches training:

```bash
cd crowd_nav

python test.py \
  --model_dir data/rgl_pledestrians_cc \
  --config data/rgl_pledestrians_cc/config.py \
  --phase test \
  -v \
  --test_case 0
```

Optional explicit overrides (should match saved config):

```bash
python test.py \
  --model_dir data/rgl_pledestrians_cc \
  --config data/rgl_pledestrians_cc/config.py \
  --phase test -v \
  --human_mix PLD_only \
  --test_scenario circle_crossing \
  --test_case 0
```

---

## Environment settings (unchanged defaults)

| Parameter | Value |
|-----------|-------|
| Episode time limit | ~49 s (`time_limit=50`, ends at `global_time >= 49`) |
| `time_step` | 0.25 s |
| Robot / human radius | 0.3 m |
| `v_pref` | 1.0 m/s |
| World (default) | 10 m × 10 m |

---

## Resume training

```bash
python train.py \
  --config data/rgl_pledestrians_cc/config.py \
  --output_dir data/rgl_pledestrians_cc \
  --resume \
  --gpu
```

---

## Related files (not modified in this change)

| File | Role |
|------|------|
| `crowd_sim/envs/policy/pledestrians.py` | PLEdestrians human policy |
| `crowd_sim/envs/policy/multi_policy.py` | Dispatches actions per human type |
| `crowd_nav/policy/gcn.py` | RGL / GCN robot policy |
| `crowd_nav/utils/explorer.py` | Episode rollouts and metrics |
| `crowd_nav/utils/trainer.py` | `VNRLTrainer` for GCN |
