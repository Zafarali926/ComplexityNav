# Reduce IL Episodes (Quick Smoke Test)

Use this to run imitation learning with **5 episodes** instead of the default **2000** (~minutes instead of hours).

---

## Recommended: override in your config only

Edit **`crowd_nav/configs/icra_benchmark/rgl.py`** — add one line inside `TrainConfig.__init__`:

```python
class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
        self.imitation_learning.il_episodes = 5   # <-- add this
```

Run as usual:

```bash
cd crowd_nav
python train.py --config configs/icra_benchmark/rgl.py --output_dir data/rgl_smoke_test --gpu
```

`train.py` reads `il_episodes` from config here:

```python
il_episodes = train_config.imitation_learning.il_episodes
explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
```

No change to `train.py` is required.

---

## Other options

| Where | What to change | Notes |
|-------|----------------|-------|
| `configs/icra_benchmark/config.py` line ~196 | `imitation_learning.il_episodes = 5` | Affects **all** configs that inherit `BaseTrainConfig` |
| `python train.py --debug ...` | Built-in | Sets IL to **10** (not 5); also shortens RL (`train_episodes = 1`) |

---

## Optional: speed up IL optimization too

With only 5 episodes, you may also lower epochs in the same `TrainConfig`:

```python
self.imitation_learning.il_epochs = 2
```

Default is **50** in `config.py` line ~198.

---

## Revert for full training

Remove the override (or set `il_episodes = 2000`) before a real benchmark run.
