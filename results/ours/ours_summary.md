# Our models

## Combined test, clean task (CER %, lower is better)

| system | ALL | dakshina_ml | sellai_manglish | sellai_mixed-code |
|---|---|---|---|---|
| Manglish-ByT5 (ours) | 2.29 | 3.18 | 1.52 | 1.26 |
| Manglish-Tiny (ours) | 2.26 | 3.19 | 1.52 | 1.17 |

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

## Robustness (CER %)

| system | base | heavy | extreme | vowel_drop | phonetic | phonetic_plus_base |
|---|---|---|---|---|---|---|
| ours_byt5 | 2.44 | 4.00 | 6.01 | 8.17 | 2.65 | 5.86 |
| ours_tiny | 2.46 | 3.94 | 5.94 | 7.23 | 2.57 | 5.00 |

## Manglish -> Malayalam script (CER / chrF)

| system | from clean | from noisy |
|---|---|---|
| rule_transliteration | 38.20 / 31.7 | 66.43 / 26.6 |
| ours_byt5 | 8.74 / 79.3 | 12.29 / 73.0 |
| ours_tiny | 7.60 / 82.2 | 11.02 / 75.9 |
