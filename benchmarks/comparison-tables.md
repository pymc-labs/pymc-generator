# Descriptive results (three seed sets)

Warm rate = accepted, validated, NPZ-persisted and read-back worlds/s, including normalization. Models reused from the one-draw grouping. Values are median [min, max], not confidence intervals.

| Workload | Ordinary warm rate | Dynamic warm rate | Specialized warm rate |
|---|---:|---:|---:|
| (5, 3) michaelis_menten / geometric | 59.05 [52.53, 63.65] | 3.63 [3.10, 4.47] | 4.96 [4.11, 5.22] |
| (5, 10) michaelis_menten / geometric | 46.78 [45.52, 48.67] | 4.24 [4.22, 4.55] | 5.02 [4.84, 5.30] |
| (15, 3) michaelis_menten / geometric | 29.17 [20.56, 31.07] | 4.16 [2.90, 4.38] | 4.86 [3.29, 5.10] |
| (15, 10) michaelis_menten / geometric | 24.87 [15.22, 26.15] | 4.25 [2.71, 4.44] | 4.98 [3.15, 5.25] |
| (10, 6) michaelis_menten / geometric | 30.54 [28.64, 36.52] | 4.02 [3.12, 4.36] | 5.02 [2.71, 5.15] |
| (10, 6) linear / geometric | 25.66 [23.84, 35.92] | 4.25 [3.01, 4.43] | 5.24 [5.15, 5.28] |
| (10, 6) hill / geometric | 28.47 [26.90, 32.46] | 4.17 [4.14, 4.34] | 4.68 [4.68, 5.04] |
| (10, 6) logistic / geometric | 31.83 [22.67, 33.38] | 4.37 [3.32, 4.40] | 5.10 [3.23, 5.21] |
| (10, 6) tanh / geometric | 32.52 [23.22, 33.63] | 4.16 [3.17, 4.33] | 4.28 [3.27, 5.22] |
| (10, 6) root / geometric | 33.33 [30.86, 40.21] | 4.20 [2.88, 4.47] | 4.89 [3.29, 5.14] |
| (10, 6) michaelis_menten / weibull | 25.65 [25.20, 28.23] | 4.18 [2.47, 4.36] | 4.69 [3.83, 4.87] |
| (10, 6) michaelis_menten / mixed | 30.33 [15.47, 38.55] | 4.26 [3.66, 4.50] | 4.75 [4.10, 5.03] |

First-use pipeline = five candidates (one per structure), including build/compile and persistence. This is NOT an independent empty-cache result per method: methods execute ordinary → dynamic → specialized in each fresh process and share that invocation's compiler cache.

| Workload | Ordinary first use (s) | Dynamic first use (s) | Specialized first use (s) |
|---|---:|---:|---:|
| (5, 3) michaelis_menten / geometric | 8.02 [7.83, 10.67] | 39.07 [34.95, 39.23] | 17.67 [14.91, 18.21] |
| (5, 10) michaelis_menten / geometric | 11.20 [10.93, 11.59] | 35.13 [34.35, 35.72] | 15.27 [14.77, 16.24] |
| (15, 3) michaelis_menten / geometric | 17.45 [17.17, 22.45] | 36.05 [35.43, 46.76] | 15.60 [15.23, 21.50] |
| (15, 10) michaelis_menten / geometric | 21.19 [20.73, 29.82] | 36.36 [34.54, 48.12] | 15.63 [14.15, 20.77] |
| (10, 6) michaelis_menten / geometric | 14.46 [13.60, 19.88] | 35.65 [34.69, 47.78] | 15.24 [15.20, 17.57] |
| (10, 6) linear / geometric | 15.62 [12.94, 16.11] | 36.77 [35.75, 44.40] | 14.70 [14.69, 15.92] |
| (10, 6) hill / geometric | 14.94 [14.73, 16.93] | 34.65 [34.54, 48.69] | 15.76 [15.74, 23.48] |
| (10, 6) logistic / geometric | 15.51 [15.30, 19.58] | 35.04 [34.97, 45.95] | 16.21 [15.70, 20.11] |
| (10, 6) tanh / geometric | 14.00 [13.86, 19.17] | 38.69 [35.98, 48.00] | 18.10 [17.23, 18.37] |
| (10, 6) root / geometric | 14.42 [14.35, 14.86] | 37.36 [35.26, 37.81] | 17.31 [15.38, 19.44] |
| (10, 6) michaelis_menten / weibull | 26.39 [25.51, 27.38] | 35.18 [34.49, 39.01] | 22.30 [22.25, 23.32] |
| (10, 6) michaelis_menten / mixed | 20.43 [18.95, 28.72] | 34.67 [34.01, 43.30] | 28.01 [26.66, 30.01] |
