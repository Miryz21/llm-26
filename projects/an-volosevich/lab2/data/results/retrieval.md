# Retrieval evaluation

Questions per language: 24. Translation of a Russian question: median 238 ms (not included in search time).

## All questions

| Experiment | Chunks | ru R@5 | ru R@10 | ru P@5 | ru P@10 | ru MRR | en R@5 | en R@10 | en P@5 | en P@10 | en MRR | Search ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 92 | 0.929 | 0.952 | 0.238 | 0.129 | 0.825 | 0.881 | 0.952 | 0.229 | 0.133 | 0.820 | 12 |
| dense | 92 | 0.881 | 0.929 | 0.219 | 0.129 | 0.744 | 0.833 | 0.929 | 0.219 | 0.129 | 0.794 | 12 |
| sparse | 92 | 0.952 | 0.952 | 0.248 | 0.124 | 0.761 | 0.929 | 0.952 | 0.238 | 0.129 | 0.827 | 11 |
| no-translation | 92 | 0.833 | 0.952 | 0.210 | 0.129 | 0.710 | 0.881 | 0.952 | 0.229 | 0.133 | 0.820 | 12 |
| no-headings | 88 | 0.762 | 0.929 | 0.200 | 0.129 | 0.693 | 0.810 | 0.929 | 0.219 | 0.133 | 0.675 | 12 |
| recursive | 63 | 0.833 | 0.976 | 0.248 | 0.152 | 0.690 | 0.833 | 0.929 | 0.248 | 0.148 | 0.712 | 12 |
| docling | 99 | 0.889 | 0.937 | 0.248 | 0.133 | 0.845 | 0.889 | 0.937 | 0.248 | 0.138 | 0.844 | 12 |
| chunk-400 | 164 | 0.897 | 0.913 | 0.248 | 0.129 | 0.721 | 0.841 | 0.913 | 0.229 | 0.129 | 0.712 | 12 |
| exact-search | 92 | 0.929 | 0.952 | 0.238 | 0.129 | 0.825 | 0.881 | 0.952 | 0.229 | 0.133 | 0.820 | 12 |
| hnsw-ef-8 | 92 | 0.929 | 0.952 | 0.238 | 0.129 | 0.825 | 0.881 | 0.952 | 0.229 | 0.133 | 0.820 | 12 |

## Formula questions (only those whose formula was parsed; the others are parsing gaps)

| Experiment | ru R@5 | ru MRR | en R@5 | en MRR | questions scored |
|---|---|---|---|---|---|
| base | 1.000 | 0.583 | 0.667 | 0.548 | 3 |
| dense | 0.667 | 0.542 | 0.667 | 0.533 | 3 |
| sparse | 1.000 | 0.567 | 1.000 | 0.567 | 3 |
| no-translation | 0.667 | 0.389 | 0.667 | 0.548 | 3 |
| no-headings | 0.333 | 0.370 | 0.333 | 0.375 | 3 |
| recursive | 0.333 | 0.245 | 0.333 | 0.208 | 3 |
| docling | 0.667 | 0.556 | 0.667 | 0.548 | 3 |
| chunk-400 | 1.000 | 0.567 | 1.000 | 0.567 | 3 |
| exact-search | 1.000 | 0.583 | 0.667 | 0.548 | 3 |
| hnsw-ef-8 | 1.000 | 0.583 | 0.667 | 0.548 | 3 |

## Parsing gaps: evidence phrases found in no chunk

| Chunk build | Question | Phrase |
|---|---|---|
| markdown-800-100-h | Q21 | `T_{\text{dist}} = \sum_{i=1}^n \|\mathbf{T}_{\text{gt}}^{i} - \mathbf{T}_{\text{gen}}^{i} \|_2` |
| markdown-800-100-h | Q23 | `s=\frac{1}{512}` |
| markdown-800-100-h | Q24 | `u'=u-\frac{W}{2}` |
| markdown-800-100-noh | Q21 | `T_{\text{dist}} = \sum_{i=1}^n \|\mathbf{T}_{\text{gt}}^{i} - \mathbf{T}_{\text{gen}}^{i} \|_2` |
| markdown-800-100-noh | Q23 | `s=\frac{1}{512}` |
| markdown-800-100-noh | Q24 | `u'=u-\frac{W}{2}` |
| recursive-800-100-h | Q21 | `T_{\text{dist}} = \sum_{i=1}^n \|\mathbf{T}_{\text{gt}}^{i} - \mathbf{T}_{\text{gen}}^{i} \|_2` |
| recursive-800-100-h | Q23 | `s=\frac{1}{512}` |
| recursive-800-100-h | Q24 | `u'=u-\frac{W}{2}` |
| docling-800-0-h | Q21 | `T_{\text{dist}} = \sum_{i=1}^n \|\mathbf{T}_{\text{gt}}^{i} - \mathbf{T}_{\text{gen}}^{i} \|_2` |
| docling-800-0-h | Q23 | `s=\frac{1}{512}` |
| docling-800-0-h | Q24 | `u'=u-\frac{W}{2}` |
| markdown-400-60-h | Q21 | `T_{\text{dist}} = \sum_{i=1}^n \|\mathbf{T}_{\text{gt}}^{i} - \mathbf{T}_{\text{gen}}^{i} \|_2` |
| markdown-400-60-h | Q23 | `s=\frac{1}{512}` |
| markdown-400-60-h | Q24 | `u'=u-\frac{W}{2}` |
