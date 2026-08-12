# Per-head gradient diagnostics

Each row measures one head on the same first training batch before the optimizer update. Adapter norms use shared LoRA and LoCon parameters; weighted norms include the configured outer head weight.

| Run | Head | Loss | Adapter norm | Weighted adapter norm | Head norm |
|---|---|---:|---:|---:|---:|
| `hda-joint_lora_gradnorm` | `hda_atac` | 4.51683 | 0.00431844 | 0.00431844 | 1.77498 |
| `hda-joint_lora_gradnorm` | `hda_rna` | 4.05731 | 0.00326696 | 0.00326696 | 2.35309 |
| `liu-hdma_lora_gradnorm` | `liu_atac` | 4.90769 | 0.00570952 | 0.00570952 | 1.50801 |
| `liu-hdma_lora_gradnorm` | `liu_rna` | 5.47416 | 0.00391946 | 0.00391946 | 2.7002 |
| `zemke2023_human_lora_gradnorm` | `zemke2023_atac` | 573.985 | 0.0555198 | 0.0555198 | 73.5069 |
| `zemke2023_human_lora_gradnorm` | `zemke2023_rna` | 13.7628 | 0.00610618 | 0.00610618 | 4.06847 |
| `zemke2023_mouse_lora_locon_organismfix_gradnorm_smoke` | `zemke2023_atac` | 2.11478 | 0.0180379 | 0.0180379 | 6.77559 |
| `zemke2023_mouse_lora_locon_organismfix_gradnorm_smoke` | `zemke2023_rna` | 1.76568 | 0.0115967 | 0.0115967 | 6.01815 |
| `zemke2023_mouse_lora_organismfix_gradnorm_smoke` | `zemke2023_atac` | 2.11351 | 0.0158915 | 0.0158915 | 6.75336 |
| `zemke2023_mouse_lora_organismfix_gradnorm_smoke` | `zemke2023_rna` | 1.76503 | 0.0106124 | 0.0106124 | 5.99879 |
| `zemke2024-all_lora_gradnorm` | `zemke2024_all_atac` | 262.402 | 0.0203578 | 0.0203578 | 11.9591 |
| `zemke2024-all_lora_gradnorm` | `zemke2024_all_rna` | 17.6637 | 0.00367935 | 0.00367935 | 3.7966 |

| Run | Head pair | Adapter-gradient cosine |
|---|---|---:|
| `hda-joint_lora_gradnorm` | `hda_atac__hda_rna` | 0.150292 |
| `liu-hdma_lora_gradnorm` | `liu_atac__liu_rna` | 0.0779338 |
| `zemke2023_human_lora_gradnorm` | `zemke2023_atac__zemke2023_rna` | 0.183041 |
| `zemke2023_mouse_lora_locon_organismfix_gradnorm_smoke` | `zemke2023_atac__zemke2023_rna` | 0.334967 |
| `zemke2023_mouse_lora_organismfix_gradnorm_smoke` | `zemke2023_atac__zemke2023_rna` | 0.27924 |
| `zemke2024-all_lora_gradnorm` | `zemke2024_all_atac__zemke2024_all_rna` | 0.132832 |
