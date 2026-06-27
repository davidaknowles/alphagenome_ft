# Low Precision LoRA Backend Comparison, 2026-06-27

This benchmark was run interactively on `ne1dg7-010.nygenome.org`, an RTX PRO
6000 Blackwell GPU node. JAX used `~/venv/jax`; PyTorch used `~/venv/torch`
through the unified `scripts/run_humanbraindev_finetune.py` launcher.

The benchmark is a short smoke-scale comparison, not a final accuracy run:

- `limit-train=4`, `limit-valid=2`, `limit-test=2`
- `batch-size=1`, `num-epochs=1`
- LoRA rank/alpha `32/32`
- Torch LoRA targets: `q_proj,v_proj`
- Torch LoCon targets: `down_blocks.4,down_blocks.5`
- GPU utilization sampled once per second with `nvidia-smi`

Raw logs and GPU samples are in `/tmp/ag_precision_lora_compare_20260627_r2`.

| Backend | Precision | Strategy | Status | Wall sec | Avg GPU % | Nonzero Avg GPU % | Max GPU % | Max VRAM MiB | Train loss | Valid loss | Test loss | Main validation metric |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| JAX | bf16 | LoRA | ok | 188.43 | 2.42 | 11.49 | 57 | 17011 | 3.2631 | 9.9807 | 21.1310 | valid R2 global = -0.0675 |
| JAX | nvfp4 | LoRA | failed | 65.57 | 3.97 | 16.12 | 53 | 4701 | NA | NA | NA | `transformer_engine[jax]` missing |
| JAX | bf16 | LoRA+LoCon | unsupported | NA | NA | NA | NA | NA | NA | NA | NA | no JAX LoCon adapter in this repo |
| JAX | nvfp4 | LoRA+LoCon | unsupported | NA | NA | NA | NA | NA | NA | NA | NA | no JAX LoCon adapter in this repo |
| Torch | bf16 | LoRA | ok | 49.62 | 1.14 | 18.67 | 29 | 17235 | 10.8964 | 45.7008 | NA | 128 bp Pearson R mean = 0.0134 |
| Torch | nvfp4 | LoRA | ok | 22.51 | 3.39 | 39.00 | 53 | 11755 | 10.8725 | 45.7382 | NA | 128 bp Pearson R mean = 0.0097 |
| Torch | bf16 | LoRA+LoCon | ok | 18.61 | 6.58 | 41.67 | 98 | 15819 | 10.8823 | 45.6774 | NA | 128 bp Pearson R mean = 0.0094 |
| Torch | nvfp4 | LoRA+LoCon | ok | 20.77 | 6.29 | 44.00 | 85 | 11887 | 10.9007 | 45.7035 | NA | 128 bp Pearson R mean = 0.0114 |

Notes:

- JAX bf16 completed and wrote train/valid/test loss and R2 metrics. Its wall
  time is dominated by fresh compile plus model setup, so whole-process average
  GPU utilization is low even though the compiled step reaches 57%.
- JAX nvfp4 fails before training in the active JAX environment:
  `ImportError: FP4 LoRA requires transformer_engine[jax]`.
- JAX LoRA+LoCon was not run because this repo currently has LoRA adapters but
  no JAX LoCon implementation.
- Torch nvfp4 LoRA reduced peak VRAM from 17.2 GiB to 11.8 GiB and cut this
  smoke benchmark wall time from 49.6 s to 22.5 s versus Torch bf16 LoRA.
- Torch LoRA+LoCon gave the highest sampled GPU peaks in this short benchmark
  and similar validation loss across bf16 and nvfp4.
