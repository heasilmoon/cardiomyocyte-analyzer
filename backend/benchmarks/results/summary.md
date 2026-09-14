
### Overall by signal mode (% of runs)

| mode        |   exact_rate |   within_1_rate |   bpm_err_pct |   n |
|:------------|-------------:|----------------:|--------------:|----:|
| consecutive |           65 |              74 |           0.4 | 360 |
| piv         |           52 |              64 |           0.8 | 360 |
| reference   |           51 |              57 |           0.3 | 360 |

### By true BPM

| mode        |   bpm |   exact_rate |   within_1_rate |   bpm_err_pct |   n |
|:------------|------:|-------------:|----------------:|--------------:|----:|
| consecutive |    30 |           75 |              93 |           0.5 |  72 |
| consecutive |    60 |           54 |              58 |           0.6 |  72 |
| consecutive |    90 |           69 |              78 |           0.2 |  72 |
| consecutive |   120 |           65 |              78 |           0.5 |  72 |
| consecutive |   180 |           61 |              62 |           0.4 |  72 |
| piv         |    30 |           54 |              74 |           1   |  72 |
| piv         |    60 |           61 |              75 |           0.8 |  72 |
| piv         |    90 |           67 |              72 |           0.3 |  72 |
| piv         |   120 |           38 |              50 |           5.6 |  72 |
| piv         |   180 |           42 |              47 |           7.3 |  72 |
| reference   |    30 |           51 |              61 |           0.5 |  72 |
| reference   |    60 |           64 |              68 |           0.2 |  72 |
| reference   |    90 |           67 |              67 |           0.2 |  72 |
| reference   |   120 |           38 |              47 |           0.6 |  72 |
| reference   |   180 |           35 |              42 |           0.3 |  72 |

### By frame rate

| mode        |   fps |   exact_rate |   within_1_rate |   bpm_err_pct |   n |
|:------------|------:|-------------:|----------------:|--------------:|----:|
| consecutive |    10 |           52 |              64 |           1.1 | 120 |
| consecutive |    15 |           65 |              75 |           0.4 | 120 |
| consecutive |    30 |           78 |              82 |           0.2 | 120 |
| piv         |    10 |           38 |              50 |           8.5 | 120 |
| piv         |    15 |           52 |              60 |           0.8 | 120 |
| piv         |    30 |           67 |              81 |           0.3 | 120 |
| reference   |    10 |           34 |              43 |           0.9 | 120 |
| reference   |    15 |           48 |              53 |           0.2 | 120 |
| reference   |    30 |           71 |              74 |           0.1 | 120 |

### By sensor noise (std, 8-bit units)

| mode        |   noise_sd |   exact_rate |   within_1_rate |   bpm_err_pct |   n |
|:------------|-----------:|-------------:|----------------:|--------------:|----:|
| consecutive |          0 |           73 |              76 |           0.3 |  90 |
| consecutive |          5 |           76 |              78 |           0.3 |  90 |
| consecutive |         10 |           71 |              76 |           0.4 |  90 |
| consecutive |         20 |           40 |              67 |           4.5 |  90 |
| piv         |          0 |           57 |              60 |           0.7 |  90 |
| piv         |          5 |           68 |              79 |           0.3 |  90 |
| piv         |         10 |           63 |              73 |           0.4 |  90 |
| piv         |         20 |           21 |              42 |          14   |  90 |
| reference   |          0 |           73 |              80 |           0.3 |  90 |
| reference   |          5 |           69 |              78 |           0.3 |  90 |
| reference   |         10 |           58 |              64 |           0.3 |  90 |
| reference   |         20 |            3 |               6 |          40.8 |  90 |

### By texture contrast (1.0 = strong speckle, 0.25 = faint)

| mode        |   texture |   exact_rate |   within_1_rate |   bpm_err_pct |   n |
|:------------|----------:|-------------:|----------------:|--------------:|----:|
| consecutive |      0.25 |           64 |              74 |           0.5 | 180 |
| consecutive |      1    |           66 |              74 |           0.4 | 180 |
| piv         |      0.25 |           49 |              59 |           1   | 180 |
| piv         |      1    |           55 |              68 |           0.7 | 180 |
| reference   |      0.25 |           49 |              56 |           0.3 | 180 |
| reference   |      1    |           52 |              58 |           0.3 | 180 |