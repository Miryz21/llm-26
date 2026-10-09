# Сводка прогонов

## По моделям

| model | mode | avg pp_tps | avg tg_tps | avg wall_s | avg quality | avg judge | avg sim |
|---|---|---|---|---|---|---|---|
| llama-3.1-8b-instruct | A_base | 4282 | 108.1 | 2.3 | 0.80 | 2.67 | 0.42 |
| llama-3.1-8b-instruct | B_tuned | 4338 | 107.1 | 1.7 | 0.85 | 2.67 | 0.71 |
| ministral-3-8b-instruct | A_base | 4376 | 98.9 | 1.6 | 0.94 | 3.67 | 0.77 |
| ministral-3-8b-instruct | B_tuned | 4371 | 98.1 | 1.6 | 0.94 | 3.67 | 0.76 |
| qwen3.8-flash-next-reap256 | A_base | 186 | 34.6 | 36.9 | 0.66 | 3.44 | 0.70 |
| qwen3.8-flash-next-reap256 | B_tuned | 187 | 34.1 | 13.1 | 0.78 | 3.44 | 0.67 |

## Детально (модель × промпт × режим)

| model | prompt | mode | n | in_tok | out_tok | reason_ch | wall_s | pp_tps | tg_tps | tok_tps | quality | judge | uniq | sim | rep3 | rep3_r | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b-instruct | P1_generation | A_base | 3 | 207 | 327.33 | 0 | 3.07 | 4182.26 | 108.25 | 210613.71 | 0.95 | 2 | 3 | 0.21 | 0.00 | 0.00 | 0 |
| llama-3.1-8b-instruct | P1_generation | B_tuned | 3 | 207 | 319 | 0 | 3.08 | 4141.82 | 105.50 | 204361.50 | 1.00 | 3 | 3 | 0.12 | 0.00 | 0.00 | 0 |
| llama-3.1-8b-instruct | P2_classification | A_base | 3 | 334 | 38 | 0 | 0.42 | 4726.18 | 108.07 | 326710.72 | 0.78 | 3 | 3 | 0.81 | 0.37 | 0.00 | 0 |
| llama-3.1-8b-instruct | P2_classification | B_tuned | 3 | 334 | 34 | 0 | 0.39 | 4744.98 | 107.74 | 271336.70 | 0.56 | 2 | 1 | 1.00 | 0.36 | 0.00 | 0 |
| llama-3.1-8b-instruct | P3_extraction | A_base | 3 | 257 | 361.67 | 0 | 3.43 | 3936.44 | 107.87 | 240367.62 | 0.67 | 3 | 3 | 0.24 | 0.03 | 0.00 | 0 |
| llama-3.1-8b-instruct | P3_extraction | B_tuned | 3 | 257 | 182 | 0 | 1.74 | 4128.08 | 108.18 | 246271.79 | 1.00 | 3 | 1 | 1.00 | 0.08 | 0.00 | 0 |
| ministral-3-8b-instruct | P1_generation | A_base | 3 | 717 | 269.33 | 0 | 2.88 | 4334.91 | 99.09 | 149631.66 | 0.81 | 3 | 3 | 0.37 | 0.00 | 0.00 | 0 |
| ministral-3-8b-instruct | P1_generation | B_tuned | 3 | 717 | 251.67 | 0 | 2.76 | 4338.74 | 96.56 | 184680.66 | 0.81 | 3 | 3 | 0.28 | 0.00 | 0.00 | 0 |
| ministral-3-8b-instruct | P2_classification | A_base | 3 | 831 | 46 | 0 | 0.64 | 4471.26 | 98.65 | 206088.05 | 1.00 | 4 | 1 | 1.00 | 0.18 | 0.00 | 0 |
| ministral-3-8b-instruct | P2_classification | B_tuned | 3 | 831 | 46 | 0 | 0.65 | 4453.52 | 98.69 | 229791.18 | 1.00 | 4 | 1 | 1.00 | 0.18 | 0.00 | 0 |
| ministral-3-8b-instruct | P3_extraction | A_base | 3 | 773 | 117 | 0 | 1.35 | 4322.48 | 98.85 | 197889.19 | 1.00 | 4 | 2 | 0.95 | 0.00 | 0.00 | 0 |
| ministral-3-8b-instruct | P3_extraction | B_tuned | 3 | 773 | 117 | 0 | 1.35 | 4320.69 | 98.95 | 203127.18 | 1.00 | 4 | 1 | 1.00 | 0.00 | 0.00 | 0 |
| qwen3.8-flash-next-reap256 | P1_generation | A_base | 3 | 209 | 3163 | 9582.33 | 92.40 | 158.29 | 34.50 | 193813.28 | 0.00 | 1 | 2 | 0.33 | 0.04 | 0.53 | 0 |
| qwen3.8-flash-next-reap256 | P1_generation | B_tuned | 3 | 209 | 656.67 | 1709.33 | 20.74 | 162.72 | 33.81 | 191936.80 | 0.33 | 1.33 | 3 | 0.02 | 0.03 | 0.14 | 1 |
| qwen3.8-flash-next-reap256 | P2_classification | A_base | 3 | 300 | 305.67 | 1186.33 | 10.30 | 217.39 | 34.35 | 210120.83 | 1.00 | 5 | 2 | 0.97 | 0.20 | 0.02 | 0 |
| qwen3.8-flash-next-reap256 | P2_classification | B_tuned | 3 | 300 | 284 | 1096 | 9.56 | 218.04 | 34.72 | 216469.05 | 1.00 | 5 | 1 | 1.00 | 0.20 | 0.01 | 0 |
| qwen3.8-flash-next-reap256 | P3_extraction | A_base | 3 | 239 | 237.67 | 448.67 | 8.13 | 182.15 | 34.81 | 194570.67 | 0.97 | 4.33 | 2 | 0.80 | 0.00 | 0.00 | 0 |
| qwen3.8-flash-next-reap256 | P3_extraction | B_tuned | 3 | 239 | 258 | 418 | 9.01 | 179.79 | 33.65 | 211410.09 | 1.00 | 4 | 1 | 1.00 | 0.00 | 0.00 | 0 |
