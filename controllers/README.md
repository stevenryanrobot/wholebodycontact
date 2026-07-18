# `controllers/` — low-level policies (the swappable base controllers)

Each subfolder is **one low-level policy** (a "controller" in the plug-and-play
sense). The force-sensing module (`forcesense/`) is controller-agnostic, so you
can swap the controller here **without retraining the sensor**.

```
controllers/
├── ceer/                     our framework's policy (GentleHumanoid fork)
│   ├── active_adaptation/    the training framework / policy source (importable
│   │                         as `active_adaptation`; installed via `pip install -e .`)
│   └── checkpoints/          trained weights + exported ONNX (gitignored, large)
├── sonic/                    (future) an external policy — self-contained plugin
│   ├── model.py              its inference network (only if not shipping ONNX)
│   ├── adapter.py            maps sim state ⇄ this policy's obs/action convention
│   ├── checkpoints/          its weights / ONNX
│   └── config.yaml           obs layout, action scale, kp/kd, reference motion
└── beyondmimic/              (future)
```

## Two kinds of policy

| | source lives in | `controllers/<name>/` holds |
|---|---|---|
| **Ours** (ceer, and its stiffness variants like `stiff30`) | `controllers/ceer/active_adaptation/` (shared) | just the trained weights under `checkpoints/` — same code, different checkpoint |
| **External** (sonic, beyondmimic) | its own `controllers/<name>/` (self-contained) | `model.py` + `adapter.py` + `checkpoints/` + `config.yaml` |

`stiff30` etc. are **not** separate policies — they are CEER checkpoints
(different kp/kd), so they live under `controllers/ceer/checkpoints/`.

## How to add a new low-level policy (e.g. Sonic)

1. `mkdir controllers/sonic/`
2. Drop in its **deployable form**: an exported `checkpoints/policy.onnx`
   (preferred — self-contained), or a PyTorch checkpoint plus a `model.py`
   that defines its network.
3. Write `adapter.py`: given the simulator's robot state, build the observation
   Sonic expects, and map Sonic's action back to joint commands (kp/kd, action
   scale, control frequency). This is the only per-policy code — a few dozen
   lines — and it is **decoupled from `forcesense/`**.
4. Write `config.yaml` recording the obs layout / action scale / gains / (if a
   tracking policy) the reference motion.

To **measure** the frozen sensor on the new controller, run a collection pass
driven by Sonic (like the cross-controller sweep in `experiments/crosspolicy/`).
To **deploy**, the residual-based sensor plugs in unchanged — no retraining,
because the residual channel is controller-invariant.
