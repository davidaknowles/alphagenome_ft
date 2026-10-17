# Agent instructions

This directory is a read-only inference bundle for a fine-tuned AlphaGenome model. `README.md` describes the model, the heads and the API.

- Do not modify, move or rewrite files here. Copy the bundle if you need a variant, and check integrity with `sha256sum -c SHA256SUMS`.
- Load the model with `alphagenome_ft.inference_bundle.load_inference_bundle`, using `alphagenome-ft` at tag `v0data-inference-v1` or later. A plain `alphagenome_ft.load_checkpoint` call fails because the heads must be registered first, and the bundle loader does this from `heads.json`.
- The pretrained AlphaGenome `all_folds` JAX weights are not in the bundle. Ask the user where their copy is rather than downloading it.
- JAX must see a GPU. On clusters, sandboxed processes may not see GPU devices; check `jax.devices()` before running long jobs.
- Read track labels from `heads.json` (`targets[*].label`) and never hardcode track indices. Track counts differ between heads, and stranded RNA heads interleave `+` and `-` channels per cell group.
- Use `head_organism_index(head)` for the organism index: 0 for human, macaque and marmoset, 1 for mouse. Give each head sequence from the assembly listed in its `sources[*].reference` entry; `provenance.reference_assemblies` in `heads.json` maps these names to assemblies.
- RNA heads were trained only at the gene level. Use `gene_expression` (exon-weighted sums of 128 bp bins) for RNA quantities, not raw per-bin values.
- Exclude RNA groups with `group_valid == False` in `genes/<head>.npz`: `L5_IT` for Zemke 2023 and `Astro1_all`, `Astro2_all`, `Micro1_all` and `Micro2_all` for Zemke 2024.
- Use 131,072 bp windows unless there is a reason to change; this is the training window. Use the same window for reference and alternate sequences when scoring variants.
- `predict_heads` evaluates all 20 heads in one forward pass and the first call compiles the model; batch sequences rather than calling it per variant with new shapes.
