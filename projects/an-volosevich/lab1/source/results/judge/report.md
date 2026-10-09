# LLM-as-a-judge (слепая оценка)


## P1_generation

| model | mode | faithfulness | completeness | tone | language | format | overall | halluc_rate |
|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b-instruct | A_base | 3.00 | 4.00 | 2.67 | 2.33 | 3.67 | 2.00 | 0.67 |
| llama-3.1-8b-instruct | B_tuned | 4.33 | 4.67 | 3.67 | 2.33 | 4.00 | 3.00 | 0.33 |
| ministral-3-8b-instruct | A_base | 4.00 | 4.00 | 4.67 | 3.67 | 4.00 | 3.00 | 0.33 |
| ministral-3-8b-instruct | B_tuned | 3.67 | 3.67 | 4.33 | 3.33 | 3.33 | 3.00 | 0.33 |
| qwen3.8-flash-next-reap256 | A_base | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen3.8-flash-next-reap256 | B_tuned | 2.33 | 2.00 | 2.00 | 1.00 | 1.33 | 1.33 | 0.00 |

## P2_classification

| model | mode | correctness | instruction_following | overall | halluc_rate |
|---|---|---|---|---|---|
| llama-3.1-8b-instruct | A_base | 3.33 | 3.00 | 3.00 | 0.67 |
| llama-3.1-8b-instruct | B_tuned | 2.00 | 2.00 | 2.00 | 1.00 |
| ministral-3-8b-instruct | A_base | 5.00 | 4.00 | 4.00 | 0.00 |
| ministral-3-8b-instruct | B_tuned | 5.00 | 4.00 | 4.00 | 0.00 |
| qwen3.8-flash-next-reap256 | A_base | 5.00 | 5.00 | 5.00 | 0.00 |
| qwen3.8-flash-next-reap256 | B_tuned | 5.00 | 5.00 | 5.00 | 0.00 |

## P3_extraction

| model | mode | accuracy | no_hallucination | instruction_following | overall | halluc_rate |
|---|---|---|---|---|---|---|
| llama-3.1-8b-instruct | A_base | 3.67 | 4.67 | 2.67 | 3.00 | 0.00 |
| llama-3.1-8b-instruct | B_tuned | 5.00 | 5.00 | 2.00 | 3.00 | 0.00 |
| ministral-3-8b-instruct | A_base | 5.00 | 5.00 | 4.00 | 4.00 | 0.00 |
| ministral-3-8b-instruct | B_tuned | 5.00 | 5.00 | 4.00 | 4.00 | 0.00 |
| qwen3.8-flash-next-reap256 | A_base | 4.67 | 5.00 | 4.67 | 4.33 | 0.00 |
| qwen3.8-flash-next-reap256 | B_tuned | 5.00 | 5.00 | 4.00 | 4.00 | 0.00 |
