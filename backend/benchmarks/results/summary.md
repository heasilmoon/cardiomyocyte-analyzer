
### Overall by signal mode (% of runs)

| mode        |   exact_rate |   within_1_rate |   bpm_err_pct |   n |
|:------------|-------------:|----------------:|--------------:|----:|
| consecutive |           84 |              92 |           0.2 | 360 |
| piv         |           69 |              84 |           0.3 | 360 |
| reference   |           61 |              68 |           0.3 | 360 |

### By true BPM

| mode        |   bpm |   exact_rate |   within_1_rate |   bpm_err_pct |   n |
|:------------|------:|-------------:|----------------:|--------------:|----:|
| consecutive |    30 |           89 |             100 |           0.2 |  72 |
| consecutive |    60 |           92 |              96 |           0.2 |  72 |
| consecutive |    90 |           92 |             100 |           0.2 |  72 |
| consecutive |   120 |           88 |             100 |           0.3 |  72 |
| consecutive |   180 |           61 |              62 |           0.4 |  72 |
| piv         |    30 |           82 |              97 |           0.3 |  72 |
| piv         |    60 |           81 |              94 |           0.6 |  72 |
| piv         |    90 |           82 |              94 |           0.2 |  72 |
| piv         |   120 |           56 |              83 |           0.3 |  72 |
| piv         |   180 |           44 |              50 |           6.2 |  72 |
| reference   |    30 |           79 |              82 |           0.3 |  72 |
| reference   |    60 |           65 |              78 |           0.2 |  72 |
| reference   |    90 |           74 |              74 |           0.3 |  72 |
| reference   |   120 |           53 |              65 |           0.5 |  72 |
| reference   |   180 |           35 |              42 |           0.8 |  72 |

### By frame rate

| mode        |   fps |   exact_rate |   within_1_rate |   bpm_err_pct |   n |
|:------------|------:|-------------:|----------------:|--------------:|----:|
| consecutive |    10 |           68 |              80 |           0.8 | 120 |
| consecutive |    15 |           90 |              97 |           0.2 | 120 |
| consecutive |    30 |           94 |              98 |           0.1 | 120 |
| piv         |    10 |           56 |              72 |           1.1 | 120 |
| piv         |    15 |           74 |              88 |           0.2 | 120 |
| piv         |    30 |           77 |              91 |           0.3 | 120 |
| reference   |    10 |           43 |              52 |           0.8 | 120 |
| reference   |    15 |           62 |              70 |           0.2 | 120 |
| reference   |    30 |           78 |              82 |           0.1 | 120 |

### By sensor noise (std, 8-bit units)

| mode        |   noise_sd |   exact_rate |   within_1_rate |   bpm_err_pct |   n |
|:------------|-----------:|-------------:|----------------:|--------------:|----:|
| consecutive |          0 |           91 |              93 |           0.2 |  90 |
| consecutive |          5 |           92 |              93 |           0.2 |  90 |
| consecutive |         10 |           91 |              93 |           0.2 |  90 |
| consecutive |         20 |           62 |              87 |           0.5 |  90 |
| piv         |          0 |           80 |              88 |           0.2 |  90 |
| piv         |          5 |           82 |              93 |           0.2 |  90 |
| piv         |         10 |           80 |              91 |           0.3 |  90 |
| piv         |         20 |           33 |              63 |           6.1 |  90 |
| reference   |          0 |           87 |              93 |           0.2 |  90 |
| reference   |          5 |           83 |              92 |           0.2 |  90 |
| reference   |         10 |           69 |              78 |           0.2 |  90 |
| reference   |         20 |            6 |               9 |          24.9 |  90 |

### By texture contrast (1.0 = strong speckle, 0.25 = faint)

| mode        |   texture |   exact_rate |   within_1_rate |   bpm_err_pct |   n |
|:------------|----------:|-------------:|----------------:|--------------:|----:|
| consecutive |      0.25 |           83 |              91 |           0.2 | 180 |
| consecutive |      1    |           85 |              92 |           0.2 | 180 |
| piv         |      0.25 |           67 |              80 |           0.4 | 180 |
| piv         |      1    |           71 |              88 |           0.3 | 180 |
| reference   |      0.25 |           58 |              66 |           0.3 | 180 |
| reference   |      1    |           64 |              70 |           0.3 | 180 |