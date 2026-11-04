# Our models

## Combined test, clean task (CER %, lower is better)

| system | ALL | dakshina_ml | sellai_manglish | sellai_mixed-code |
|---|---|---|---|---|
| Manglish-ByT5 (ours) | 2.29 | 3.18 | 1.52 | 1.26 |
| Manglish-Tiny (ours) | 2.26 | 3.19 | 1.52 | 1.17 |
| Manglish-ByT5 + English (ours) | 2.36 | 3.27 | 1.56 | 1.31 |
| Manglish-Tiny + English (ours) | 2.31 | 3.18 | 1.56 | 1.30 |
| Manglish-Tiny, script-focused (ours) | 2.20 | 3.03 | 1.55 | 1.22 |

Paired bootstrap, CER(ours) - CER(other), negative favours ours:

- ours_byt5 - byt5_plusall: -1.26 (95% CI -1.44 to -1.08)
- ours_byt5 - mbart_plusall: -3.67 (95% CI -3.86 to -3.46)
- ours_byt5 - mt5_lr1e-3_plusall: -4.15 (95% CI -4.37 to -3.95)
- ours_byt5 - byt5: -13.71 (95% CI -14.74 to -12.69)
- ours_byt5 - mbart: -12.63 (95% CI -13.00 to -12.31)
- ours_byt5 - rule_collapse_runs: -8.69 (95% CI -8.87 to -8.49)
- ours_tiny - byt5_plusall: -1.28 (95% CI -1.45 to -1.12)
- ours_tiny - mbart_plusall: -3.70 (95% CI -3.89 to -3.52)
- ours_tiny - mt5_lr1e-3_plusall: -4.18 (95% CI -4.40 to -3.96)
- ours_tiny - byt5: -13.71 (95% CI -14.66 to -12.73)
- ours_tiny - mbart: -12.67 (95% CI -13.01 to -12.35)
- ours_tiny - rule_collapse_runs: -8.72 (95% CI -8.90 to -8.53)
- ours_byt5_en - byt5_plusall: -1.18 (95% CI -1.34 to -1.01)
- ours_byt5_en - mbart_plusall: -3.59 (95% CI -3.81 to -3.40)
- ours_byt5_en - mt5_lr1e-3_plusall: -4.08 (95% CI -4.30 to -3.87)
- ours_byt5_en - byt5: -13.61 (95% CI -14.64 to -12.56)
- ours_byt5_en - mbart: -12.57 (95% CI -12.90 to -12.25)
- ours_byt5_en - rule_collapse_runs: -8.62 (95% CI -8.80 to -8.42)
- ours_tiny_en - byt5_plusall: -1.24 (95% CI -1.41 to -1.07)
- ours_tiny_en - mbart_plusall: -3.64 (95% CI -3.85 to -3.44)
- ours_tiny_en - mt5_lr1e-3_plusall: -4.13 (95% CI -4.36 to -3.88)
- ours_tiny_en - byt5: -13.68 (95% CI -14.68 to -12.66)
- ours_tiny_en - mbart: -12.60 (95% CI -12.96 to -12.28)
- ours_tiny_en - rule_collapse_runs: -8.67 (95% CI -8.86 to -8.48)
- ours_tiny_script - byt5_plusall: -1.35 (95% CI -1.54 to -1.18)
- ours_tiny_script - mbart_plusall: -3.76 (95% CI -3.95 to -3.55)
- ours_tiny_script - mt5_lr1e-3_plusall: -4.25 (95% CI -4.46 to -4.04)
- ours_tiny_script - byt5: -13.80 (95% CI -14.84 to -12.78)
- ours_tiny_script - mbart: -12.72 (95% CI -13.09 to -12.40)
- ours_tiny_script - rule_collapse_runs: -8.78 (95% CI -8.96 to -8.60)

## Robustness (CER %)

| system | base | heavy | extreme | vowel_drop | phonetic | phonetic_plus_base |
|---|---|---|---|---|---|---|
| ours_byt5 | 2.44 | 4.00 | 6.01 | 8.17 | 2.65 | 5.86 |
| ours_tiny | 2.46 | 3.94 | 5.94 | 7.23 | 2.57 | 5.00 |
| ours_byt5_en | 2.54 | 4.14 | 7.02 | 8.57 | 2.73 | 5.32 |
| ours_tiny_en | 2.44 | 4.15 | 7.38 | 8.49 | 2.59 | 5.27 |
| ours_tiny_script | 2.36 | 3.91 | 6.32 | 7.86 | 2.44 | 4.91 |

## Manglish -> Malayalam script (CER / chrF)

| system | from clean | from noisy |
|---|---|---|
| rule_transliteration | 38.20 / 31.7 | 66.43 / 26.6 |
| ours_byt5 | 8.74 / 79.3 | 12.29 / 73.0 |
| ours_tiny | 7.60 / 82.2 | 11.02 / 75.9 |
| ours_byt5_en | 9.76 / 80.0 | 12.68 / 74.8 |
| ours_tiny_en | 7.43 / 83.9 | 10.56 / 78.2 |
| ours_tiny_script | 7.51 / 85.0 | 10.14 / 80.2 |
