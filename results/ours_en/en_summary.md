# English and code-mixed script

## Manglish -> English, in22_conv (BLEU / chrF, higher is better)

| system | roman | noisy | mixed |
|---|---|---|---|
| Manglish-ByT5 script, then NLLB-200 | 18.2 / 43.1 | 14.7 / 38.4 | 17.8 / 42.8 |
| Manglish-ByT5 + English script, then NLLB-200 | 22.0 / 47.1 | 18.2 / 42.8 | 20.8 / 46.0 |
| Manglish-ByT5 + English (ours), direct | 9.3 / 33.3 | 8.6 / 31.9 | 9.8 / 33.7 |
| Manglish-Tiny + English script, then NLLB-200 | 22.1 / 47.6 | 18.8 / 43.3 | 21.5 / 47.2 |
| Manglish-Tiny + English (ours), direct | 4.1 / 27.2 | 3.8 / 26.2 | 4.3 / 27.5 |
| NLLB-200 on the real Malayalam (ceiling) | 26.3 / 50.8 (any input) | | |

Paired bootstrap on mean sentence chrF, direct model minus pipeline (positive favours ours):

- roman: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -9.52 (95% CI -10.54 to -8.56)
- roman: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -13.54 (95% CI -14.43 to -12.63)
- roman: Manglish-ByT5 + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -13.93 (95% CI -14.86 to -13.04)
- roman: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -15.56 (95% CI -16.50 to -14.63)
- roman: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -19.58 (95% CI -20.47 to -18.62)
- roman: Manglish-Tiny + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -19.97 (95% CI -20.90 to -18.98)
- noisy: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -6.62 (95% CI -7.62 to -5.74)
- noisy: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -10.72 (95% CI -11.60 to -9.75)
- noisy: Manglish-ByT5 + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -11.46 (95% CI -12.41 to -10.55)
- noisy: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -12.00 (95% CI -12.96 to -11.06)
- noisy: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -16.11 (95% CI -17.08 to -15.12)
- noisy: Manglish-Tiny + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -16.85 (95% CI -17.78 to -15.92)
- mixed: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -8.94 (95% CI -9.82 to -7.99)
- mixed: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -12.37 (95% CI -13.27 to -11.49)
- mixed: Manglish-ByT5 + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -13.15 (95% CI -14.02 to -12.26)
- mixed: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -15.08 (95% CI -16.06 to -14.12)
- mixed: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -18.52 (95% CI -19.50 to -17.57)
- mixed: Manglish-Tiny + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -19.30 (95% CI -20.27 to -18.32)

## Manglish -> English, in22_gen (BLEU / chrF, higher is better)

| system | roman | noisy | mixed |
|---|---|---|---|
| Manglish-ByT5 script, then NLLB-200 | 13.2 / 38.5 | 11.1 / 35.4 | 12.8 / 37.8 |
| Manglish-ByT5 + English script, then NLLB-200 | 15.8 / 40.8 | 13.7 / 38.1 | 14.9 / 40.3 |
| Manglish-ByT5 + English (ours), direct | 7.8 / 30.7 | 6.3 / 28.0 | 8.1 / 31.1 |
| Manglish-Tiny + English script, then NLLB-200 | 16.4 / 42.3 | 12.5 / 36.8 | 15.8 / 41.9 |
| Manglish-Tiny + English (ours), direct | 3.1 / 25.7 | 2.5 / 23.6 | 3.5 / 26.4 |
| NLLB-200 on the real Malayalam (ceiling) | 30.5 / 58.5 (any input) | | |

Paired bootstrap on mean sentence chrF, direct model minus pipeline (positive favours ours):

- roman: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -9.13 (95% CI -9.86 to -8.39)
- roman: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -11.95 (95% CI -12.71 to -11.09)
- roman: Manglish-ByT5 + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -13.30 (95% CI -14.09 to -12.52)
- roman: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -14.98 (95% CI -15.77 to -14.19)
- roman: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -17.81 (95% CI -18.60 to -17.03)
- roman: Manglish-Tiny + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -19.16 (95% CI -20.01 to -18.27)
- noisy: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -8.03 (95% CI -8.77 to -7.33)
- noisy: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -11.33 (95% CI -12.05 to -10.57)
- noisy: Manglish-ByT5 + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -10.21 (95% CI -10.92 to -9.50)
- noisy: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -13.45 (95% CI -14.21 to -12.72)
- noisy: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -16.74 (95% CI -17.53 to -15.96)
- noisy: Manglish-Tiny + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -15.62 (95% CI -16.39 to -14.85)
- mixed: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -7.98 (95% CI -8.80 to -7.22)
- mixed: Manglish-ByT5 + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -10.75 (95% CI -11.58 to -9.96)
- mixed: Manglish-ByT5 + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -12.38 (95% CI -13.11 to -11.61)
- mixed: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 script, then NLLB-200: -13.65 (95% CI -14.47 to -12.84)
- mixed: Manglish-Tiny + English (ours), direct vs Manglish-ByT5 + English script, then NLLB-200: -16.42 (95% CI -17.20 to -15.61)
- mixed: Manglish-Tiny + English (ours), direct vs Manglish-Tiny + English script, then NLLB-200: -18.04 (95% CI -18.85 to -17.24)

## Code-mixed Manglish -> Malayalam script (807 of 2527 sentences have English words)

| system | English words translated | left in Latin | CER / chrF (mixed) | CER / chrF (no English words) |
|---|---|---|---|---|
| Manglish-ByT5 (before English) | 14.9% | 0.0% | 28.58 / 65.4 | 27.18 / 67.1 |
| Manglish-ByT5 + English (ours), direct | 24.3% | 0.0% | 25.30 / 71.9 | 23.96 / 73.6 |
| Manglish-Tiny (before English) | 16.9% | 0.0% | 33.86 / 61.1 | 32.49 / 62.7 |
| Manglish-Tiny + English (ours) | 35.3% | 0.2% | 22.18 / 74.5 | 21.27 / 75.8 |
