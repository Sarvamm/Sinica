# Sinica 

[![PyPI version](https://badge.fury.io/py/sinica.svg)](https://badge.fury.io/py/sinica)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Sinica** is a lightweight, fast, and intuitive Python data profiling tool built on top of `pandas`. It provides comprehensive statistical summaries and automated data quality alerts with zero bloated dependencies.

---

##  Key Features

- **Instant Statistical Summaries**: Get detailed, column-level statistical insights (mean, median, mode, missing values, min/max) in a single method call.
- **Automated Quality Alerts**: Detect data hygiene issues automatically, including:
  - Constant or zero-filled columns
  - High correlation between features
  - High cardinality & severe class imbalance
  - Skewed distributions & missing data percentages
  - Duplicate rows & empty dataset checks
- ⚡ **Lightweight & Clean Output**: Returns clean, standard `pandas.DataFrame` objects that seamlessly integrate with your analytical workflows or reporting pipelines.

---

##  Installation

Install `sinica` directly from PyPI using `pip`:

```bash
pip install sinica
```

#  Quick Start

Here is how you can profile a dataset in under 10 seconds:
```python
import pandas as pd
from sinica import DataProfiler

# 1. Load your DataFrame
df = your_data_frame 

# 2. Instantiate the DataProfiler
pr = DataProfiler(df)

# 3. Generate a statistical summary
summary_df = pr.summary()
print(summary_df)

# 4. Extract data quality alerts
alerts_df = pr.alerts()
print(alerts_df)
```

