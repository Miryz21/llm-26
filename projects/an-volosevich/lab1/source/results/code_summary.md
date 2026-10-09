# Coding suite (English)

## Ranking (B_tuned, greedy decoding)

| rank | model | problems solved (of 8) | tests passed | A_base tests passed |
|---|---|---|---|---|
| 1 | ministral-3-8b-instruct | 7 | 91% | 77% |
| 2 | qwen3.8-flash-next-reap256 | 6 | 75% | 88% |
| 3 | llama-3.1-8b-instruct | 5 | 72% | 78% |

## Per problem

| model | problem | mode | tests passed | solved | out_tok | wall_s |
|---|---|---|---|---|---|---|
| qwen3.8-flash-next-reap256 | C1_palindrome | A_base | 100% | 100% | 99 | 4.0 |
| qwen3.8-flash-next-reap256 | C1_palindrome | B_tuned | 100% | 100% | 101 | 4.0 |
| qwen3.8-flash-next-reap256 | C2_merge_intervals | A_base | 100% | 100% | 207 | 7.2 |
| qwen3.8-flash-next-reap256 | C2_merge_intervals | B_tuned | 100% | 100% | 243 | 8.1 |
| qwen3.8-flash-next-reap256 | C3_int_to_roman | A_base | 100% | 100% | 5167 | 151.1 |
| qwen3.8-flash-next-reap256 | C3_int_to_roman | B_tuned | 0% | 0% | 4143 | 117.9 |
| qwen3.8-flash-next-reap256 | C3_int_to_roman | B_long | 100% | 100% | 1109 | 33.6 |
| qwen3.8-flash-next-reap256 | C4_top_k_words | A_base | 100% | 100% | 176 | 6.3 |
| qwen3.8-flash-next-reap256 | C4_top_k_words | B_tuned | 100% | 100% | 155 | 5.6 |
| qwen3.8-flash-next-reap256 | C5_decode_string | A_base | 100% | 100% | 476 | 14.8 |
| qwen3.8-flash-next-reap256 | C5_decode_string | B_tuned | 100% | 100% | 504 | 15.6 |
| qwen3.8-flash-next-reap256 | C6_longest_valid_parens | A_base | 100% | 100% | 252 | 8.3 |
| qwen3.8-flash-next-reap256 | C6_longest_valid_parens | B_tuned | 100% | 100% | 343 | 11.0 |
| qwen3.8-flash-next-reap256 | C7_min_window | A_base | 100% | 100% | 419 | 13.2 |
| qwen3.8-flash-next-reap256 | C7_min_window | B_tuned | 100% | 100% | 887 | 26.4 |
| qwen3.8-flash-next-reap256 | C8_calculator | A_base | 0% | 0% | 16208 | 469.7 |
| qwen3.8-flash-next-reap256 | C8_calculator | B_tuned | 0% | 0% | 8192 | 236.8 |
| qwen3.8-flash-next-reap256 | C8_calculator | B_long | 0% | 0% | 32768 | 967.1 |
| llama-3.1-8b-instruct | C1_palindrome | A_base | 100% | 100% | 46 | 0.4 |
| llama-3.1-8b-instruct | C1_palindrome | B_tuned | 100% | 100% | 43 | 0.4 |
| llama-3.1-8b-instruct | C2_merge_intervals | A_base | 100% | 100% | 213 | 1.9 |
| llama-3.1-8b-instruct | C2_merge_intervals | B_tuned | 100% | 100% | 115 | 1.0 |
| llama-3.1-8b-instruct | C3_int_to_roman | A_base | 100% | 100% | 141 | 1.2 |
| llama-3.1-8b-instruct | C3_int_to_roman | B_tuned | 100% | 100% | 152 | 1.3 |
| llama-3.1-8b-instruct | C4_top_k_words | A_base | 80% | 0% | 69 | 0.6 |
| llama-3.1-8b-instruct | C4_top_k_words | B_tuned | 40% | 0% | 62 | 0.6 |
| llama-3.1-8b-instruct | C5_decode_string | A_base | 100% | 100% | 130 | 1.2 |
| llama-3.1-8b-instruct | C5_decode_string | B_tuned | 17% | 0% | 131 | 1.2 |
| llama-3.1-8b-instruct | C6_longest_valid_parens | A_base | 100% | 100% | 128 | 1.1 |
| llama-3.1-8b-instruct | C6_longest_valid_parens | B_tuned | 100% | 100% | 176 | 1.5 |
| llama-3.1-8b-instruct | C7_min_window | A_base | 43% | 0% | 240 | 2.1 |
| llama-3.1-8b-instruct | C7_min_window | B_tuned | 100% | 100% | 234 | 2.1 |
| llama-3.1-8b-instruct | C8_calculator | A_base | 0% | 0% | 324 | 2.8 |
| llama-3.1-8b-instruct | C8_calculator | B_tuned | 18% | 0% | 472 | 4.1 |
| ministral-3-8b-instruct | C1_palindrome | A_base | 100% | 100% | 46 | 0.6 |
| ministral-3-8b-instruct | C1_palindrome | B_tuned | 100% | 100% | 52 | 0.6 |
| ministral-3-8b-instruct | C2_merge_intervals | A_base | 100% | 100% | 147 | 1.5 |
| ministral-3-8b-instruct | C2_merge_intervals | B_tuned | 100% | 100% | 133 | 1.4 |
| ministral-3-8b-instruct | C3_int_to_roman | A_base | 100% | 100% | 180 | 1.8 |
| ministral-3-8b-instruct | C3_int_to_roman | B_tuned | 100% | 100% | 190 | 1.9 |
| ministral-3-8b-instruct | C4_top_k_words | A_base | 100% | 100% | 93 | 1.0 |
| ministral-3-8b-instruct | C4_top_k_words | B_tuned | 100% | 100% | 89 | 1.0 |
| ministral-3-8b-instruct | C5_decode_string | A_base | 17% | 0% | 138 | 1.4 |
| ministral-3-8b-instruct | C5_decode_string | B_tuned | 100% | 100% | 128 | 1.3 |
| ministral-3-8b-instruct | C6_longest_valid_parens | A_base | 100% | 100% | 90 | 1.0 |
| ministral-3-8b-instruct | C6_longest_valid_parens | B_tuned | 100% | 100% | 90 | 1.0 |
| ministral-3-8b-instruct | C7_min_window | A_base | 100% | 100% | 255 | 2.5 |
| ministral-3-8b-instruct | C7_min_window | B_tuned | 100% | 100% | 251 | 2.5 |
| ministral-3-8b-instruct | C8_calculator | A_base | 0% | 0% | 562 | 5.5 |
| ministral-3-8b-instruct | C8_calculator | B_tuned | 27% | 0% | 410 | 4.0 |
