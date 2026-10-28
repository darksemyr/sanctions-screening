# Notebooks

For looking at things, not for producing results.

Name them `NN-initials-topic.ipynb`, e.g. `01-vs-first-look.ipynb`, so two
people do not both create `Untitled.ipynb` and collide in git.

If something here turns into a result that belongs in the report, move the
code into `src/screening/` and run it from a script. Notebooks explore,
scripts produce.

Clear all outputs before committing (Kernel > Restart & Clear Output).
Committed output cells make every diff unreadable.

Start here:

```python
import sys; sys.path.insert(0, "../src")
import pandas as pd
from screening import config as C

df = pd.read_parquet(C.PAIRS_SCOPED)
df.head()
```
