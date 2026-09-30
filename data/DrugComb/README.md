# DrugComb data

Supplied study files:

- `DrugComb_6x6_complete_blocks_36points.csv.gz`
- `DrugComb_6x6_CRISPR_TopVar1000.csv`
- `DrugComb_10x10_SAFIR_input.csv.gz`
- `DrugComb_10x10_SAFIR_input_CRISPR8.csv`

The two largest CSV files were gzip-compressed without changing their tabular contents. Pandas reads them directly:

```python
import pandas as pd
df = pd.read_csv("DrugComb_6x6_complete_blocks_36points.csv.gz")
```
