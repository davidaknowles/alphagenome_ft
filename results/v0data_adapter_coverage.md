# Canonical adapter coverage

The primary table covers each requested non-ENCODE study. Cross-species studies additionally require evaluation of both joint adapter checkpoints against every native species. A matched result means both strategies completed the same epoch; it does not imply that early stopping completed or that the requested correlation was reached.

## Primary studies

| Study | Canonical comparison | Native species required | Missing native evaluations | Coverage status |
|---|---|---|---|---|
| Mannens HDA | `hda-joint` | not applicable |  | comparison coverage complete |
| Johansen 2025 | `johansen_joint` | human, macaque, marmoset | lora: human, macaque, marmoset; lora+locon: human, macaque, marmoset | missing lora, lora+locon |
| Liu HDMA | `liu-hdma` | not applicable |  | comparison coverage complete |
| Zemke 2023 | `zemke2023_joint` | human, macaque, marmoset, mouse | lora: human, macaque, marmoset; lora+locon: human, macaque, marmoset, mouse | missing native evaluations |
| Zemke 2024 | `zemke2024-all` | not applicable |  | comparison coverage complete |

## All canonical arms

| Dataset | Latest LoRA epoch | Latest LoRA+LoCon epoch | Highest matched epoch | Status |
|---|---:|---:|---:|---|
| `hda` | 4 | 4 | 4 | matched result available |
| `hda-joint` | 3 | 2 | 2 | matched result available |
| `johansen-human` | 3 | 2 | 2 | matched result available |
| `johansen_joint` |  |  |  | missing lora, lora+locon |
| `liu-hdma` | 1 | 1 | 1 | matched result available |
| `zemke2023-human` | 4 | 4 | 4 | matched result available |
| `zemke2023_macaque` | 1 | 1 | 1 | matched result available |
| `zemke2023_marmoset` | 1 | 1 | 1 | matched result available |
| `zemke2023_mouse` |  |  |  | missing lora, lora+locon |
| `zemke2023_joint` | 1 | 1 | 1 | matched result available |
| `zemke2024-all` | 5 | 5 | 5 | matched result available |
