# Agent Instructions

- If `hostname` is `ne1-login` or similar, this is a login node; submit compute-intensive jobs with Slurm.
- Otherwise, this is a compute node and tests/GPU smoke checks can run directly here.
- On GPU compute nodes, sandboxed commands may not see `/dev/nvidia*`; run GPU diagnostics and GPU smoke checks outside the sandbox when CUDA/JAX reports no device despite being on a GPU node.
- Use `~/venv/jax` for JAX backend work and `~/venv/torch` for PyTorch backend work.
