| system | seeds | CER | WER | BLEU | chrF |
|---|---|---|---|---|---|
| mBART-50 | 5 | 7.91 ± 0.23 | 25.02 ± 0.32 | 51.68 ± 0.54 | 84.40 ± 0.37 |
| ByT5-base | 5 | 6.58 ± 1.63 | 23.07 ± 2.39 | 56.76 ± 2.46 | 88.35 ± 1.52 |
| mT5-base (LR 1e-3, deviation) | 3 | 10.02 ± 0.47 | 30.40 ± 1.41 | 42.19 ± 2.40 | 82.07 ± 0.80 |
| mT5-base (LR 2e-4, as specified) | 1 | 47.35 | 73.60 | 7.99 | 46.64 |
| mBART-50 + Mixed-Code + Dakshina | 1 | 5.06 | 16.51 | 65.67 | 89.52 |
| ByT5-base + Mixed-Code + Dakshina | 1 | 2.37 | 14.13 | 69.09 | 93.31 |
| mT5-base + Mixed-Code + Dakshina (LR 2e-4) | 1 | 8.43 | 30.06 | 41.89 | 83.03 |
| mT5-base + Mixed-Code + Dakshina (LR 1e-3) | 1 | 6.19 | 20.23 | 58.00 | 87.69 |

Paired bootstrap on CER (seed 42, negative favours the first system):

- mbart_large - byt5_base: -0.54 (95% CI -2.48 to +1.26)
- mbart_large - mt5_base_lr1e-3: -2.84 (95% CI -3.85 to -1.72)
- byt5_base - mt5_base_lr1e-3: -2.30 (95% CI -4.31 to -0.15)
- mbart_large_plusall - mbart_large: -2.49 (95% CI -3.17 to -1.78)
- byt5_base_plusall - byt5_base: -5.71 (95% CI -7.63 to -4.03)
- byt5_base_plusall - mbart_large_plusall: -2.68 (95% CI -3.42 to -1.97)

Same comparisons on the combined test set (seed 42, negative favours the first system):

- mbart - byt5: -1.07 (95% CI -2.03 to -0.06, n=2059)
- mbart_plusall - mbart: -8.96 (95% CI -9.32 to -8.63, n=2059)
- byt5_plusall - byt5: -12.45 (95% CI -13.58 to -11.37, n=2059)
- byt5_plusall - mbart_plusall: -2.42 (95% CI -2.65 to -2.20, n=2059)
- mt5_lr1e-3_plusall - mt5_lr1e-3: -19.41 (95% CI -21.07 to -17.79, n=2059)
- byt5_plusall - mt5_lr1e-3_plusall: -2.89 (95% CI -3.15 to -2.59, n=2059)