n_cpu_moe (from --fit): 43
| model                          |       size |     params | backend    | ngl |  n_cpu_moe | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------: | --: | --------------: | -------------------: |
| qwen4exp A3B Q3_K - Medium     |  57.68 GiB |   116.51 B | CUDA       |  99 |         43 |       8 |   1 |           pp256 |        399.72 ± 0.00 |
| qwen4exp A3B Q3_K - Medium     |  57.68 GiB |   116.51 B | CUDA       |  99 |         43 |       8 |   1 |            tg64 |         30.95 ± 0.00 |
| qwen4exp A3B Q3_K - Medium     |  57.68 GiB |   116.51 B | CUDA       |  99 |         43 |      12 |   1 |           pp256 |        404.51 ± 0.00 |
| qwen4exp A3B Q3_K - Medium     |  57.68 GiB |   116.51 B | CUDA       |  99 |         43 |      12 |   1 |            tg64 |         33.28 ± 0.00 |
| qwen4exp A3B Q3_K - Medium     |  57.68 GiB |   116.51 B | CUDA       |  99 |         43 |      16 |   1 |           pp256 |        405.88 ± 0.00 |
| qwen4exp A3B Q3_K - Medium     |  57.68 GiB |   116.51 B | CUDA       |  99 |         43 |      16 |   1 |            tg64 |         32.07 ± 0.00 |

build: 86a283532 (11524)
