# VBOC — Viability-Based Optimal Control

Data generation and neural network training pipeline to approximate the **viability kernel** of the **aSTedH** platform via VBOC (Viability-Based Optimal Control).

## Usage

```bash
python main.py --system sth [--generation] [--training] [--plot] [--check] [--epochs N] [--horizon N]
```

| Flag | Description |
|---|---|
| `--generation` | Run VBOC data generation (parallelised) |
| `--training` | Train the neural network on generated data |
| `--plot` | Save diagnostic plots |
| `--check` | Single deterministic solve for debugging |
| `--horizon N` | Override the default prediction horizon |

All parameters are in the `Parameters` class (`vboc/parser.py`).

## Pipeline

1. **Data generation** — solves VBOC OCPs from randomised orientations and obstacle boxes; results saved to `data/`.
2. **Training** — fits a feedforward network mapping `[box, orientation, velocity direction] → max safe speed`; checkpoint saved to `nn/`.
3. **Plotting** — fixed-direction viability kernel sections and training diagnostics.

## Requirements

`acados`, `casadi`, `torch`, `numpy`, `scipy`, `adam-robotics`, `urdf_parser_py`, `rich`, `tqdm`, `matplotlib`

---

Copyright © 2025. All rights reserved.