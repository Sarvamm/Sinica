import re
import unicodedata
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from pandas import DataFrame
from scipy import stats
from tqdm import tqdm
from wordcloud import WordCloud


class ProfileReport:
    """A tool for profiling and generating statistical summaries of pandas DataFrames."""

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        pd.set_option("future.no_silent_downcasting", True)
        self.df = df

        # Initialize core attributes
        self.num_vars: int = 0
        self.num_obs: int = 0
        self.total_cells: int = 0
        self.missing_cells_count: int = 0
        self.missing_pct: float = 0.0
        self.dup_rows: int = 0
        self.dup_pct: float = 0.0

        # Initialize dataframes & analysis outputs
        self.basic_stats_df: pd.DataFrame = pd.DataFrame()
        self.dtypes: pd.Series = pd.Series(dtype=object)
        self.dtype_counts_df: pd.DataFrame = pd.DataFrame()
        self.summary_df: pd.DataFrame = pd.DataFrame()

        self.missing_cols: list[str] = []
        self.missing_df: pd.DataFrame = pd.DataFrame()
        self.missing_corr: pd.DataFrame = pd.DataFrame()

        # Run profiling pipeline
        self._profile_dataframe()

    def _is_hashable(self, val) -> bool:
        """Helper to check if a value or column contents can be hashed."""
        try:
            hash(val)
            return True
        except TypeError:
            return False

    def _get_hashable_subset(self) -> list[str]:
        """Returns columns that can be safely used for duplicate checks."""
        valid_cols = []
        for col in self.df.columns:
            sample = self.df[col].dropna()
            if not sample.empty:
                if self._is_hashable(sample.iloc[0]):
                    valid_cols.append(col)
            else:
                valid_cols.append(col)
        return valid_cols

    def _profile_dataframe(self) -> None:
        """Executes initialization steps sequentially with progress bar feedback."""
        with tqdm(total=4, desc="Understanding DataFrame") as pbar:
            # Step 1: Basic Dimensions & Row Stats
            pbar.set_description("Calculating dataset dimensions & missing statistics")
            self._compute_basic_stats()
            pbar.update(1)

            # Step 2: Data Types Breakdown
            pbar.set_description("Analyzing data types")
            self._compute_dtype_counts()
            pbar.update(1)

            # Step 3: Column Summary Table
            pbar.set_description("Building statistical summary table")
            self._compute_summary()
            pbar.update(1)

            # Step 4: Missing Value Breakdown & Correlations
            pbar.set_description("Analyzing missing value distributions")
            self._compute_missing_analysis()
            pbar.update(1)

    def _compute_basic_stats(self) -> None:
        self.num_vars = self.df.shape[1]
        self.num_obs = self.df.shape[0]
        self.total_cells = self.num_vars * self.num_obs

        self.missing_cells_count = int(self.df.isna().sum().sum())
        self.missing_pct = (
            (self.missing_cells_count / self.total_cells) * 100
            if self.total_cells > 0
            else 0.0
        )

        hashable_cols = self._get_hashable_subset()
        if hashable_cols:
            self.dup_rows = int(self.df.duplicated(subset=hashable_cols).sum())
        else:
            self.dup_rows = 0

        self.dup_pct = (self.dup_rows / self.num_obs) * 100 if self.num_obs > 0 else 0.0

        self.basic_stats_df = pd.DataFrame(
            {
                "Metric": [
                    "Number of variables",
                    "Number of observations",
                    "Missing cells",
                    "Missing cells (%)",
                    "Duplicate rows",
                    "Duplicate rows (%)",
                ],
                "Value": [
                    self.num_vars,
                    self.num_obs,
                    self.missing_cells_count,
                    f"{self.missing_pct:.1f}%",
                    self.dup_rows,
                    f"{self.dup_pct:.1f}%",
                ],
            }
        )

    def _compute_dtype_counts(self) -> None:
        self.dtypes = self.df.dtypes.astype(str)
        self.dtype_counts_df = self.dtypes.value_counts().reset_index()
        self.dtype_counts_df.columns = ["Data Type", "Count"]

    def _compute_summary(self) -> None:
        # Convert unhashable/complex columns to string representation safely
        df_eval = self.df.copy()
        for col in df_eval.columns:
            if df_eval[col].dtype == "object":
                sample = df_eval[col].dropna()
                if not sample.empty and not self._is_hashable(sample.iloc[0]):
                    df_eval[col] = df_eval[col].apply(
                        lambda x: str(x) if isinstance(x, (dict, list)) else x
                    )

        df_description: DataFrame = df_eval.describe(include="all").T
        summary: DataFrame = df_description.drop(
            columns=["25%", "75%"], errors="ignore"
        )
        summary["dtype"] = self.dtypes
        # Use df_eval.nunique() instead of self.df.nunique() to avoid dict hash crashes
        summary["unique"] = df_eval.nunique(dropna=False)
        summary["missing%"] = (self.df.isna().mean() * 100).round(2)

        front_cols: list[str] = ["dtype", "count", "unique", "missing%"]
        other_cols: list[str] = [c for c in summary.columns if c not in front_cols]

        self.summary_df = (
            summary[front_cols + other_cols].sort_values(by="dtype").fillna("-").T
        )

    def _compute_missing_analysis(self) -> None:
        self.missing_cols = self.df.columns[self.df.isna().any()].tolist()

        missing_counts = pd.DataFrame(
            self.df.isna().sum().sort_values(ascending=False),
            columns=["missing_count"],
        )
        missing_counts["missing_pct"] = (
            (missing_counts["missing_count"] / self.num_obs * 100).round(2)
            if self.num_obs > 0
            else 0.0
        )
        self.missing_df = missing_counts

        if self.missing_cols:
            hashable_missing = [
                c for c in self.missing_cols if c in self._get_hashable_subset()
            ]
            if len(hashable_missing) > 1:
                self.missing_corr = self.df[hashable_missing].isna().corr()
            else:
                self.missing_corr = pd.DataFrame()
        else:
            self.missing_corr = pd.DataFrame()

    def show_overview(self) -> None:
        """Renders the complete data overview directly in Jupyter Notebooks."""
        display(Markdown("# Data Overview"))
        display(Markdown("## Basic stats:"))
        display(self.get_basic_stats())
        display(Markdown("## Variable types:"))
        display(self.get_dtype_counts())
        display(Markdown("## Data Alerts"))
        display(self.get_alerts())
        display(Markdown("## Data Summary:"))
        display(self.get_summary())
        self.show_missing_report()

        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                numeric_df.corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
            )
            plt.title("Correlation heatmap")
            plt.tight_layout()
            plt.show()

    def _ipython_display_(self) -> None:
        self.show_overview()

    def get_basic_stats(self) -> pd.DataFrame:
        return self.basic_stats_df

    def get_dtype_counts(self) -> pd.DataFrame:
        return self.dtype_counts_df

    def get_summary(self) -> pd.DataFrame:
        return self.summary_df

    def show_missing_report(self) -> None:
        if self.missing_cells_count == 0:
            return

        display(Markdown("# Missing value report"))
        display(
            Markdown(
                f"Total number of missing values: {self.missing_cells_count}"
                f" ({self.missing_pct:.1f}%)"
            )
        )
        display(Markdown(" - Variables with missing values:"))
        display(self.missing_cols)
        display(Markdown(" - Missing count table"))
        display(self.missing_df)

        if not self.missing_df.empty and self.missing_cells_count > 0:
            plt.title("Percentage of Missing Values by variable")
            sns.barplot(
                data=self.missing_df.missing_pct.sort_values(ascending=False),
                orient="h",
            )
            plt.xlabel("Percentage of Missing Values")
            plt.show()

            plt.title("Missing Value matrix")
            sns.heatmap(self.df.isna().T, cbar=False)
            plt.show()

            if not self.missing_corr.empty and len(self.missing_cols) > 1:
                plt.title("Missing Value Correlation")
                mask = np.triu(np.ones_like(self.missing_corr, dtype=bool))
                sns.heatmap(
                    self.missing_corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    cbar=True,
                    mask=mask,
                )
                plt.show()

    def get_alerts(
        self,
        correlation_threshold: float = 0.8,
        high_cardinality_threshold: int = 50,
        imbalance_threshold: float = 0.8,
        skewness_threshold: float = 1.0,
        uniform_pvalue_threshold: float = 0.999,
        duplicate_threshold: int = 10,
    ) -> pd.DataFrame:
        alerts_list: list[dict[str, str]] = []

        if self.df.empty:
            return pd.DataFrame(
                [{"Description": "Dataset is empty", "Alert Type": "Empty"}]
            )

        if self.dup_rows > duplicate_threshold:
            alerts_list.append(
                {
                    "Description": (
                        f"Dataset has {self.dup_rows} ({self.dup_pct:.1f}%) duplicate"
                        " rows"
                    ),
                    "Alert Type": "Duplicates",
                }
            )

        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().abs()
            for col in corr_matrix.columns:
                high_corr_cols = corr_matrix.index[
                    (corr_matrix[col] >= correlation_threshold)
                    & (corr_matrix.index != col)
                ].tolist()

                if high_corr_cols:
                    main_target = high_corr_cols[0]
                    other_count = len(high_corr_cols) - 1
                    other_text = (
                        f" and {other_count} other fields" if other_count > 0 else ""
                    )
                    alerts_list.append(
                        {
                            "Description": (
                                f"{col} is highly correlated with"
                                f" {main_target}{other_text}"
                            ),
                            "Alert Type": "High correlation",
                        }
                    )

        for col in self.df.columns:
            s = self.df[col]
            clean_s = s.dropna()
            n_unique = clean_s.nunique()
            n_missing = s.isnull().sum()

            if n_unique == 1:
                val = clean_s.iloc[0] if not clean_s.empty else ""
                alerts_list.append(
                    {
                        "Description": f'{col} has constant value "{val}"',
                        "Alert Type": "Constant",
                    }
                )
                if val == 0:
                    alerts_list.append(
                        {
                            "Description": f"{col} contains only zeros",
                            "Alert Type": "Zeros",
                        }
                    )
            elif n_unique > high_cardinality_threshold:
                alerts_list.append(
                    {
                        "Description": (
                            f"{col} has a high cardinality: {n_unique} distinct values"
                        ),
                        "Alert Type": "High cardinality",
                    }
                )

            sample_val = clean_s.iloc[0] if not clean_s.empty else None
            if not clean_s.empty and n_unique > 1 and self._is_hashable(sample_val):
                try:
                    top_freq = clean_s.value_counts(normalize=True).iloc[0]
                    if top_freq >= imbalance_threshold:
                        pct = round(top_freq * 100, 1)
                        alerts_list.append(
                            {
                                "Description": (
                                    f"{col} is highly imbalanced ({pct}% top value)"
                                ),
                                "Alert Type": "Imbalance",
                            }
                        )
                except Exception:
                    pass

            if n_missing > 0:
                pct = round((n_missing / self.num_obs) * 100, 1)
                alerts_list.append(
                    {
                        "Description": f"{col} has {n_missing} ({pct}%) missing values",
                        "Alert Type": "Missing",
                    }
                )

            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                inf_count = np.isinf(s).sum()
                if inf_count > 0:
                    alerts_list.append(
                        {
                            "Description": f"{col} has {inf_count} infinite values",
                            "Alert Type": "Infinite",
                        }
                    )

            if (
                n_unique == self.num_obs
                and self.num_obs > 0
                and self._is_hashable(sample_val)
            ):
                alerts_list.append(
                    {
                        "Description": f"{col} has all unique values",
                        "Alert Type": "Unique",
                    }
                )

            if (
                pd.api.types.is_numeric_dtype(s)
                and not pd.api.types.is_bool_dtype(s)
                and len(clean_s) > 3
            ):
                col_skew = abs(clean_s.skew())
                if col_skew >= skewness_threshold:
                    alerts_list.append(
                        {
                            "Description": f"{col} is highly skewed ({round(col_skew, 2)})",
                            "Alert Type": "Skewed",
                        }
                    )

                try:
                    counts, _ = np.histogram(clean_s, bins=10)
                    _, p_val = stats.chisquare(counts)
                    if p_val >= uniform_pvalue_threshold:
                        alerts_list.append(
                            {
                                "Description": f"{col} follows a uniform distribution",
                                "Alert Type": "Uniform",
                            }
                        )
                except Exception:
                    pass

        if not alerts_list:
            return pd.DataFrame(columns=["Description", "Alert Type"])

        return pd.DataFrame(alerts_list)

    def text_univ(self, col: str):
        """Computes comprehensive text statistics matching ydata-profiling,

        displays length, unicode, sample, and word frequency tables, and plots a word
        cloud.
        """
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        n = len(self.df)
        original_col = self.df[col]
        clean_s = original_col.dropna().astype(str)

        # --- 1. Basic Metrics ---
        distinct = clean_s.nunique()
        distinct_pct = (distinct / n * 100) if n > 0 else 0.0

        missing = original_col.isna().sum()
        missing_pct = (missing / n * 100) if n > 0 else 0.0

        mem_bytes = original_col.memory_usage(deep=True)
        if mem_bytes < 1024:
            mem_str = f"{mem_bytes} B"
        elif mem_bytes < 1024 * 1024:
            mem_str = f"{mem_bytes / 1024:.1f} KiB"
        else:
            mem_str = f"{mem_bytes / (1024 * 1024):.1f} MiB"

        # --- 2. Length Statistics ---
        lengths = clean_s.str.len()
        max_len = lengths.max() if not lengths.empty else 0
        min_len = lengths.min() if not lengths.empty else 0
        mean_len = lengths.mean() if not lengths.empty else 0.0
        median_len = lengths.median() if not lengths.empty else 0.0

        # --- 3. Characters & Unicode Statistics ---
        all_text = "".join(clean_s.tolist())
        total_chars = len(all_text)
        distinct_chars = len(set(all_text))
        distinct_categories = len({unicodedata.category(c) for c in all_text})
        distinct_scripts = 1 if all_text else 0
        distinct_blocks = 1 if all_text else 0

        # --- 4. Word Frequency Statistics (Top 5 & Bottom 5) ---
        words = re.findall(r"\b\w+\b", all_text.lower())
        word_counts = Counter(words)
        most_common = word_counts.most_common(5)

        # Sort by frequency ascending, then alphabetically for bottom 5
        sorted_words = sorted(word_counts.items(), key=lambda x: (x[1], x[0]))
        bottom_common = sorted_words[:5]

        # --- 5. Sample Rows (First 5 rows) ---
        sample_rows = clean_s.head(5).tolist()
        while len(sample_rows) < 5:
            sample_rows.append("-")

        # --- Display Header and Tables ---
        display(Markdown(f"### **{col}**"))
        display(Markdown("*Text*"))

        # Overview Table
        display(Markdown("#### Overview"))
        overview_table = pd.DataFrame(
            {
                "Metric": [
                    "Distinct",
                    "Distinct (%)",
                    "Missing",
                    "Missing (%)",
                    "Memory size",
                ],
                "Value": [
                    distinct,
                    f"{distinct_pct:.1f}%",
                    missing,
                    f"{missing_pct:.1f}%",
                    mem_str,
                ],
            }
        )
        display(overview_table)

        # Length Statistics Table
        display(Markdown("#### Length statistics"))
        length_table = pd.DataFrame(
            {
                "Length Metric": [
                    "Max length",
                    "Median length",
                    "Mean length",
                    "Min length",
                ],
                "Value": [max_len, median_len, f"{mean_len:.4f}", min_len],
            }
        )
        display(length_table)

        # Characters and Unicode Table
        display(Markdown("#### Characters and Unicode"))
        unicode_table = pd.DataFrame(
            {
                "Characters and Unicode": [
                    "Total characters",
                    "Distinct characters",
                    "Distinct categories",
                    "Distinct scripts",
                    "Distinct blocks",
                ],
                "Value": [
                    f"{total_chars:,}",
                    distinct_chars,
                    distinct_categories,
                    distinct_scripts,
                    distinct_blocks,
                ],
            }
        )
        display(unicode_table)

        # Unique Table
        display(Markdown("#### Unique statistics"))
        unique_table = pd.DataFrame(
            {
                "Unique": ["Unique", "Unique (%)"],
                "Value": [distinct, f"{distinct_pct:.1f}%"],
            }
        )
        display(unique_table)

        # Sample Table
        display(Markdown("#### Sample rows"))
        sample_table = pd.DataFrame(
            {
                "Sample": ["1st row", "2nd row", "3rd row", "4th row", "5th row"],
                "Value": sample_rows,
            }
        )
        display(sample_table)

        # Word Frequency Tables
        display(Markdown("#### Word Frequency Analysis"))
        top_words_df = pd.DataFrame(
            most_common, columns=["Most Frequent Word", "Count"]
        )
        bottom_words_df = pd.DataFrame(
            bottom_common, columns=["Least Frequent Word", "Count"]
        )

        display(Markdown("**Top 5 Most Frequent Words:**"))
        display(top_words_df)

        display(Markdown("**Bottom 5 Least Frequent Words:**"))
        display(bottom_words_df)

        # --- 6. WordCloud Generation ---
        if all_text.strip():
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="Dark2",
                max_words=150,
                contour_width=1,
            ).generate(all_text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud of {col}", fontsize=14, fontweight="bold", pad=15)
            plt.tight_layout()
            plt.show()

    def numeric_univ(self, col: str):
        """Computes basic, quantile, and descriptive statistics matching ydata-profiling,

        displays the metrics tables, and plots both a histogram and a boxplot.
        """
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        s = self.df[col]
        n = len(self.df)
        clean_s = s.dropna()

        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            raise TypeError(f"Column '{col}' must be a continuous numeric type.")

        # --- 1. Basic Metrics Calculation ---
        distinct = clean_s.nunique()
        distinct_pct = (distinct / n * 100) if n > 0 else 0.0

        missing = s.isna().sum()
        missing_pct = (missing / n * 100) if n > 0 else 0.0

        infinite = np.isinf(s).sum()
        infinite_pct = (infinite / n * 100) if n > 0 else 0.0

        zeros = (s == 0).sum()
        zeros_pct = (zeros / n * 100) if n > 0 else 0.0

        negative = (s < 0).sum()
        negative_pct = (negative / n * 100) if n > 0 else 0.0

        mean_val = clean_s.mean() if not clean_s.empty else np.nan
        min_val = clean_s.min() if not clean_s.empty else np.nan
        max_val = clean_s.max() if not clean_s.empty else np.nan

        mem_bytes = s.memory_usage(deep=True)
        if mem_bytes < 1024:
            mem_str = f"{mem_bytes} B"
        elif mem_bytes < 1024 * 1024:
            mem_str = f"{mem_bytes / 1024:.1f} KiB"
        else:
            mem_str = f"{mem_bytes / (1024 * 1024):.1f} MiB"

        # --- 2. Quantile Statistics Calculation ---
        q1 = clean_s.quantile(0.25) if not clean_s.empty else np.nan
        median_val = clean_s.median() if not clean_s.empty else np.nan
        q3 = clean_s.quantile(0.75) if not clean_s.empty else np.nan
        p5 = clean_s.quantile(0.05) if not clean_s.empty else np.nan
        p95 = clean_s.quantile(0.95) if not clean_s.empty else np.nan
        iqr = q3 - q1 if not (pd.isna(q3) or pd.isna(q1)) else np.nan
        rng = (
            max_val - min_val if not (pd.isna(max_val) or pd.isna(min_val)) else np.nan
        )

        # --- 3. Descriptive Statistics Calculation ---
        std_val = clean_s.std() if not clean_s.empty else np.nan
        cv_val = (
            std_val / abs(mean_val)
            if mean_val != 0 and not np.isnan(mean_val)
            else np.nan
        )
        kurt_val = clean_s.kurt() if not clean_s.empty else np.nan
        skew_val = clean_s.skew() if not clean_s.empty else np.nan
        sum_val = clean_s.sum() if not clean_s.empty else np.nan
        var_val = clean_s.var() if not clean_s.empty else np.nan
        mad_val = (clean_s - median_val).abs().median() if not clean_s.empty else np.nan

        # Monotonicity check
        is_inc = s.is_monotonic_increasing
        is_dec = s.is_monotonic_decreasing
        if is_inc and is_dec:
            monotonicity = "Constant"
        elif is_inc:
            monotonicity = "Monotonic increasing"
        elif is_dec:
            monotonicity = "Monotonic decreasing"
        else:
            monotonicity = "Not monotonic"

        # --- Display Header and Tables ---
        display(Markdown(f"### **{col}**"))
        display(Markdown(f"*Real number (ℝ)* — DataType: `{s.dtype}`"))

        # Overview Table
        display(Markdown("#### Overview statistics"))
        basic_table = pd.DataFrame(
            {
                "Metric (Left)": [
                    "Distinct",
                    "Distinct (%)",
                    "Missing",
                    "Missing (%)",
                    "Infinite",
                    "Infinite (%)",
                    "Mean",
                ],
                "Value (Left)": [
                    distinct,
                    f"{distinct_pct:.1f}%",
                    missing,
                    f"{missing_pct:.1f}%",
                    infinite,
                    f"{infinite_pct:.1f}%",
                    f"{mean_val:.4f}" if not np.isnan(mean_val) else "-",
                ],
                "Metric (Right)": [
                    "Minimum",
                    "Maximum",
                    "Zeros",
                    "Zeros (%)",
                    "Negative",
                    "Negative (%)",
                    "Memory size",
                ],
                "Value (Right)": [
                    min_val,
                    max_val,
                    zeros,
                    f"{zeros_pct:.1f}%",
                    negative,
                    f"{negative_pct:.1f}%",
                    mem_str,
                ],
            }
        )
        display(basic_table)

        # Quantile & Descriptive Side-by-Side Tables
        display(Markdown("#### Quantile & Descriptive statistics"))
        quantile_table = pd.DataFrame(
            {
                "Quantile statistic": [
                    "Minimum",
                    "5-th percentile",
                    "Q1",
                    "median",
                    "Q3",
                    "95-th percentile",
                    "Maximum",
                    "Range",
                    "Interquartile range (IQR)",
                ],
                "Value": [min_val, p5, q1, median_val, q3, p95, max_val, rng, iqr],
            }
        )

        desc_table = pd.DataFrame(
            {
                "Descriptive statistic": [
                    "Standard deviation",
                    "Coefficient of variation (CV)",
                    "Kurtosis",
                    "Mean",
                    "Median Absolute Deviation (MAD)",
                    "Skewness",
                    "Sum",
                    "Variance",
                    "Monotonicity",
                ],
                "Value": [
                    f"{std_val:.5f}" if not np.isnan(std_val) else "-",
                    f"{cv_val:.8f}" if not np.isnan(float(cv_val)) else "-",
                    f"{kurt_val:.7f}" if not np.isnan(kurt_val) else "-",
                    f"{mean_val:.3f}" if not np.isnan(mean_val) else "-",
                    f"{mad_val}" if not np.isnan(float(mad_val)) else "-",
                    f"{skew_val:.8f}" if not np.isnan(skew_val) else "-",
                    f"{sum_val:.6e}" if abs(sum_val) > 1e6 else f"{sum_val}",
                    f"{var_val:.6e}" if abs(var_val) > 1e6 else f"{var_val}",
                    monotonicity,
                ],
            }
        )

        # Display them as side-by-side markdown tables or separate tables
        display(quantile_table)
        display(desc_table)

        # --- Plotting Histogram & Boxplot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        sns.histplot(data=self.df, x=col, ax=axes[0], kde=True, color="dodgerblue")
        axes[0].set_title(f"Histogram of {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")

        sns.boxplot(data=self.df, x=col, ax=axes[1], color="lightgreen")
        axes[1].set_title(f"Boxplot of {col}")
        axes[1].set_xlabel(col)

        plt.tight_layout()
        plt.show()

    def date_univ(self, col: str):
        """Computes date/datetime statistics matching ydata-profiling,

        displays the metrics tables, and plots a time-series histogram.
        """
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        n = len(self.df)
        original_col = self.df[col]

        # Safely parse column into datetimes, capturing invalid/NaT entries
        parsed_dt = pd.to_datetime(original_col, errors="coerce")
        invalid_count = original_col.notna().sum() - parsed_dt.notna().sum()
        invalid_pct = (invalid_count / n * 100) if n > 0 else 0.0

        clean_s = parsed_dt.dropna()

        # --- 1. Calculate Metrics Matching Image ---
        distinct = clean_s.nunique()
        distinct_pct = (distinct / n * 100) if n > 0 else 0.0

        missing = original_col.isna().sum() + invalid_count
        missing_pct = (missing / n * 100) if n > 0 else 0.0

        min_val = clean_s.min() if not clean_s.empty else "-"
        max_val = clean_s.max() if not clean_s.empty else "-"

        # Memory size formatting
        mem_bytes = original_col.memory_usage(deep=True)
        if mem_bytes < 1024:
            mem_str = f"{mem_bytes} B"
        elif mem_bytes < 1024 * 1024:
            mem_str = f"{mem_bytes / 1024:.1f} KiB"
        else:
            mem_str = f"{mem_bytes / (1024 * 1024):.1f} MiB"

        # --- 2. Display Header and Tables ---
        display(Markdown(f"### **{col}**"))
        display(Markdown("*Date*"))

        # Optional: Display 'Unique' badge if all valid values are distinct
        if distinct == len(clean_s) and n > 0:
            display(
                Markdown(
                    "<span"
                    " style='background-color:#d9534f; color:white; padding:2px 8px;"
                    " border-radius:4px; font-weight:bold; font-size:12px;'>Unique</span>"
                )
            )

        # Left & Right metric mappings
        left_metrics = {
            "Distinct": distinct,
            "Distinct (%)": f"{distinct_pct:.1f}%",
            "Missing": missing,
            "Missing (%)": f"{missing_pct:.1f}%",
            "Memory size": mem_str,
        }

        right_metrics = {
            "Minimum": str(min_val),
            "Maximum": str(max_val),
            "Invalid dates": invalid_count,
            "Invalid dates (%)": f"{invalid_pct:.1f}%",
        }

        # Build side-by-side or stacked summary view
        left_table = pd.DataFrame(
            {
                "Metric (Left)": list(left_metrics.keys()),
                "Value (Left)": list(left_metrics.values()),
            }
        )
        right_table = pd.DataFrame(
            {
                "Metric (Right)": list(right_metrics.keys()),
                "Value (Right)": list(right_metrics.values()),
            }
        )

        display(Markdown("#### Overview statistics"))
        display(left_table)
        display(right_table)

        # --- 3. Plotting Date Histogram ---
        if not clean_s.empty:
            plt.figure(figsize=(10, 4))
            sns.histplot(x=parsed_dt, kde=False, color="dodgerblue", bins=50)
            plt.title(f"Histogram of {col}", fontsize=14, fontweight="bold", pad=15)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Warning: Column '{col}' contains no valid dates to plot.")

    def categorical_univ(self, col: str):
        """Computes categorical statistics matching ydata-profiling,

        displays overview and common values tables, and plots both the frequency bar
        chart and the segmented Common Values proportional bar plot.
        """
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        n = len(self.df)
        original_col = self.df[col]
        clean_s = original_col.dropna()

        # --- 1. Calculate Metrics ---
        distinct = clean_s.nunique()
        distinct_pct = (distinct / n * 100) if n > 0 else 0.0

        missing = original_col.isna().sum()
        missing_pct = (missing / n * 100) if n > 0 else 0.0

        # Memory size formatting
        mem_bytes = original_col.memory_usage(deep=True)
        if mem_bytes < 1024:
            mem_str = f"{mem_bytes} B"
        elif mem_bytes < 1024 * 1024:
            mem_str = f"{mem_bytes / 1024:.1f} KiB"
        else:
            mem_str = f"{mem_bytes / (1024 * 1024):.1f} MiB"

        val_counts = clean_s.value_counts()
        val_freqs = clean_s.value_counts(normalize=True) * 100

        # --- 2. Display Header and Overview Table ---
        display(Markdown(f"### **{col}**"))
        display(Markdown("*Categorical*"))

        overview_metrics = {
            "Distinct": distinct,
            "Distinct (%)": f"{distinct_pct:.1f}%",
            "Missing": missing,
            "Missing (%)": f"{missing_pct:.1f}%",
            "Memory size": mem_str,
        }

        overview_table = pd.DataFrame(
            {
                "Metric": list(overview_metrics.keys()),
                "Value": list(overview_metrics.values()),
            }
        )

        display(Markdown("#### Overview statistics"))
        display(overview_table)

        # --- 3. Plotting Segmented "Common Values (Plot)" Bar Chart ---
        if not clean_s.empty:
            fig, ax = plt.subplots(figsize=(9, 3.5))

            # Take top 5 categories for the proportional bar plot (matching standard profiles)
            plot_data = val_counts.head(5)
            total_valid = clean_s.count()

            left = 0
            colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]

            for i, (cat, count) in enumerate(plot_data.items()):
                pct = (count / total_valid) * 100
                width = count
                color = colors[i % len(colors)]

                ax.barh(
                    0,
                    width,
                    left=left,
                    color=color,
                    label=str(cat),
                    edgecolor="white",
                    linewidth=1.5,
                )

                # Text inside bar segment
                x_center = left + width / 2
                ax.text(
                    x_center,
                    0,
                    f"{pct:.1f}%\n({count})",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=11,
                )

                left += width

            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(
                "Common Values (Plot)",
                fontsize=14,
                fontweight="bold",
                loc="left",
                pad=15,
            )

            # Hide box spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Legend placed below
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(0.15, -0.15),
                ncol=1,
                frameon=True,
                fontsize=12,
            )
            plt.tight_layout()
            plt.show()

        # --- 4. Common Values Table ---
        display(Markdown("#### Common Values"))
        common_values_df = pd.DataFrame(
            {
                "Value": val_counts.index,
                "Count": val_counts.values,
                "Frequency (%)": [f"{f:.1f}%" for f in val_freqs.values],
            }
        )

        display(common_values_df.head(10))

    def univ(self, col: str, col_type: str):
        """Univariate analysis router function requiring an explicit col_type argument.

        Supported col_types: 'numeric', 'text', 'date', 'categorical' (with common
        aliases).
        """
        # 1. Validate DataFrame input
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("Input 'df' must be a valid pandas DataFrame.")

        # 2. Validate column existence
        if col not in self.df.columns:
            raise ValueError(
                f"Column '{col}' does not exist in the provided DataFrame."
            )

        # 3. Validate explicit col_type input
        if col_type is None:
            raise ValueError(
                f"Missing required argument 'col_type' for column '{col}'. Please "
                "explicitly specify one of: 'numeric', 'text', 'date', 'categorical'."
            )

        col_type_norm = str(col_type).lower().strip()

        # 4. Route to the selected analysis method and report failures.
        try:
            if col_type_norm in ("numeric", "num", "int", "float"):
                self.numeric_univ(col)
            elif col_type_norm in ("text", "string", "str"):
                self.text_univ(col)
            elif col_type_norm in ("date", "datetime", "timestamp"):
                self.date_univ(col)
            elif col_type_norm in ("categorical", "category", "cat"):
                self.categorical_univ(col)
            else:
                raise ValueError(
                    f"Unsupported col_type '{col_type}'. Supported types: 'numeric',"
                    " 'text', 'date', 'categorical'."
                )
        except Exception as e:
            display(
                Markdown(
                    f"<span style='color:red;'>❌ **Error profiling column '{col}'"
                    f" as '{col_type}':** {str(e)}</span>"
                )
            )
            raise e
