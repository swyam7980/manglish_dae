# Second-stage translators (BLEU / chrF, beam 5)

## in22_conv

| system | licence | real Malayalam | Tiny, typed | Tiny, noisy | Tiny, mixed | ByT5, typed | ByT5, noisy | tiny_dialect_dialect | tiny_dialect_mixed | tiny_dialect_noisy | tiny_dialect_typed | tiny_script_dialect | tiny_script_mixed | tiny_script_noisy | tiny_script_typed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| it2_dist | MIT | 30.5 / 55.1 | 25.9 / 51.6 | 21.2 / 46.9 | 25.4 / 51.1 | 25.7 / 51.2 | 21.1 / 46.7 | - | - | - | - | - | - | - | - |
| it2_1b | MIT | 31.2 / 55.4 | 26.6 / 52.0 | 22.5 / 47.7 | 26.2 / 51.5 | 25.8 / 51.2 | 21.8 / 47.0 | 27.3 / 52.6 | 27.3 / 52.4 | 24.1 / 49.2 | 27.6 / 52.8 | 25.9 / 51.4 | 27.1 / 52.4 | 23.4 / 48.7 | 27.3 / 52.6 |
| opus | Apache-2.0 | 13.8 / 34.7 | 10.8 / 32.2 | 8.9 / 29.6 | 10.6 / 31.8 | 10.4 / 31.7 | 8.1 / 29.1 | - | - | - | - | - | - | - | - |
| nllb | CC BY-NC 4.0 | 26.9 / 51.5 | 23.3 / 48.6 | 19.5 / 44.1 | 22.6 / 48.0 | 22.7 / 48.1 | 19.0 / 44.0 | - | - | - | - | - | - | - | - |

## in22_gen

| system | licence | real Malayalam | Tiny, typed | Tiny, noisy | Tiny, mixed | ByT5, typed | ByT5, noisy | tiny_dialect_dialect | tiny_dialect_mixed | tiny_dialect_noisy | tiny_dialect_typed | tiny_script_dialect | tiny_script_mixed | tiny_script_noisy | tiny_script_typed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| it2_dist | MIT | 38.2 / 64.6 | 20.6 / 46.5 | 15.7 / 40.2 | 19.7 / 45.9 | 20.1 / 45.7 | 16.9 / 41.8 | - | - | - | - | - | - | - | - |
| it2_1b | MIT | 40.2 / 66.1 | 21.9 / 47.7 | 16.7 / 41.1 | 21.1 / 47.1 | 21.5 / 46.7 | 18.0 / 42.9 | 23.1 / 49.0 | 22.6 / 48.8 | 18.4 / 43.0 | 23.5 / 49.2 | 23.1 / 48.7 | 22.3 / 47.9 | 18.0 / 42.6 | 23.2 / 48.7 |
| opus | Apache-2.0 | 5.8 / 28.6 | 4.7 / 23.9 | 3.8 / 21.3 | 4.3 / 23.4 | 4.8 / 23.3 | 3.9 / 21.8 | - | - | - | - | - | - | - | - |
| nllb | CC BY-NC 4.0 | 31.6 / 59.4 | 17.4 / 43.5 | 13.2 / 37.7 | 16.9 / 43.1 | 16.8 / 42.0 | 14.5 / 39.0 | - | - | - | - | - | - | - | - |

Paired bootstrap on sentence chrF (first minus second, positive favours the first):

- in22_conv/gold: it2_1b - it2_dist: +0.22 (95% CI -0.31 to +0.76)
- in22_conv/tiny_typed: it2_1b - it2_dist: +0.13 (95% CI -0.45 to +0.70)
- in22_conv/gold: it2_1b - opus: +16.90 (95% CI +15.80 to +17.97)
- in22_conv/tiny_typed: it2_1b - opus: +16.09 (95% CI +14.87 to +17.16)
- in22_conv/gold: it2_1b - nllb: +3.41 (95% CI +2.60 to +4.20)
- in22_conv/tiny_typed: it2_1b - nllb: +2.33 (95% CI +1.54 to +3.08)
- in22_conv/typed: it2_1b from byt5 - it2_1b from tiny: -0.46 (95% CI -0.91 to -0.01)
- in22_conv/dialect: it2_1b from tiny_dialect - it2_1b from tiny_script: +1.20 (95% CI +0.81 to +1.66)
- in22_conv/typed: it2_1b from tiny_dialect - it2_1b from tiny: +0.93 (95% CI +0.47 to +1.41)
- in22_conv/typed: it2_1b from tiny_script - it2_1b from tiny: +0.57 (95% CI +0.18 to +0.96)
- in22_gen/gold: it2_1b - it2_dist: +1.43 (95% CI +0.96 to +1.87)
- in22_gen/tiny_typed: it2_1b - it2_dist: +1.26 (95% CI +0.85 to +1.70)
- in22_gen/gold: it2_1b - opus: +36.35 (95% CI +35.42 to +37.31)
- in22_gen/tiny_typed: it2_1b - opus: +26.61 (95% CI +25.70 to +27.54)
- in22_gen/gold: it2_1b - nllb: +6.55 (95% CI +5.89 to +7.24)
- in22_gen/tiny_typed: it2_1b - nllb: +4.58 (95% CI +3.90 to +5.17)
- in22_gen/typed: it2_1b from byt5 - it2_1b from tiny: -0.83 (95% CI -1.26 to -0.37)
- in22_gen/dialect: it2_1b from tiny_dialect - it2_1b from tiny_script: +0.27 (95% CI -0.02 to +0.56)
- in22_gen/typed: it2_1b from tiny_dialect - it2_1b from tiny: +1.50 (95% CI +1.15 to +1.87)
- in22_gen/typed: it2_1b from tiny_script - it2_1b from tiny: +1.17 (95% CI +0.81 to +1.51)
