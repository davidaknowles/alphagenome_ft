# Canonical adapter coverage

A matched result means both strategies completed the same epoch. It does not imply that early stopping completed or that the requested correlation was reached.

| Dataset | Latest LoRA epoch | Latest LoRA+LoCon epoch | Highest matched epoch | Status |
|---|---:|---:|---:|---|
| `hda` | 4 | 4 | 4 | matched result available |
| `hda-joint` | 3 | 2 | 2 | matched result available |
| `johansen-human` |  |  |  | missing lora, lora+locon |
| `johansen_joint` |  |  |  | missing lora, lora+locon |
| `liu-hdma` | 1 |  |  | missing lora+locon |
| `zemke2023-human` | 4 | 4 | 4 | matched result available |
| `zemke2023_macaque` | 1 | 1 | 1 | matched result available |
| `zemke2023_marmoset` | 1 | 1 | 1 | matched result available |
| `zemke2023_mouse` | 1 | 1 | 1 | matched result available |
| `zemke2023_joint` | 1 | 1 | 1 | matched result available |
| `zemke2024-all` | 5 | 5 | 5 | matched result available |
