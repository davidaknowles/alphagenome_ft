# JAX/Torch Backend Pipeline

The HumanBrainDev launcher now has a backend switch:

```bash
python scripts/run_humanbraindev_finetune.py --backend jax ...
python scripts/run_humanbraindev_finetune.py --backend torch --backbone-lora ...
```

Shared responsibilities in this repository:

- BigWig discovery and target head setup.
- Chromosome/fold/BED split generation.
- Optional windowed target-cache build for JAX runs.
- Backend-neutral R2 metric semantics in `alphagenome_ft.finetune.metrics`.
- Backend dispatch types in `alphagenome_ft.finetune.backends`.

Backend responsibilities:

- `jax`: uses the existing optimized JAX training loop, target cache, LoRA adapter
  checkpointing, and JAX/Transformer Engine low-precision options.
- `torch`: writes the shared splits as BED files and delegates model training to
  `../alphagenome-pytorch/scripts/finetune.py`, preserving PyTorch low-precision
  options such as `nvfp8` and `nvfp4`.

The current Torch adapter is intentionally a subprocess boundary. This keeps the
PyTorch model implementation, adapter injection, delta checkpoints, and torchao
low-precision paths in the PyTorch repo while allowing this repo to own the
shared data/split surface. The next step is to move PyTorch evaluation onto the
shared R2 metric contract so Torch and JAX runs emit directly comparable
train/valid/test metrics.
