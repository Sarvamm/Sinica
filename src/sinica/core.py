import numpy as np
import pandas as pd
from scipy import stats


class ProfileReport:
    """
    A tool for profiling and generating statistical summaries of pandas DataFrames.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df

    def summary(self, decimals: int = 4) -> pd.DataFrame:
        """Generates a complete column-by-column statistical summary."""
        _info = []

        for col in self.df.columns:
            s = self.df[col]
            clean_s = s.dropna()
            mode_val = clean_s.mode()

            entry = {
                "Column": str(col),
                "dtype": str(s.dtype),
                "missing": int(s.isnull().sum()),
                "unique": int(s.nunique()),
            }

            if pd.api.types.is_numeric_dtype(s):
                entry.update(
                    {
                        "mean": round(float(clean_s.mean()), decimals)
                        if not clean_s.empty
                        else None,
                        "median": round(float(clean_s.median()), decimals)
                        if not clean_s.empty
                        else None,
                        "mode": round(float(mode_val.iloc[0]), decimals)
                        if not mode_val.empty
                        else None,
                        "min": round(float(clean_s.min()), decimals)
                        if not clean_s.empty
                        else None,
                        "max": round(float(clean_s.max()), decimals)
                        if not clean_s.empty
                        else None,
                    }
                )
            else:
                entry.update(
                    {
                        "mean": "-",
                        "median": "-",
                        "mode": str(mode_val.iloc[0]) if not mode_val.empty else "-",
                        "min": str(clean_s.min()) if not clean_s.empty else "-",
                        "max": str(clean_s.max()) if not clean_s.empty else "-",
                    }
                )

            _info.append(entry)

        return pd.DataFrame(_info).set_index("Column")

    def missing_report(self) -> pd.DataFrame:
        """Returns a breakdown of missing values per column with percentages."""
        total = self.df.isnull().sum()
        percent = (total / len(self.df)) * 100
        report = pd.DataFrame(
            {"Missing Count": total, "Percentage (%)": round(percent, 2)}
        )
        return report[report["Missing Count"] > 0].sort_values(
            by="Missing Count", ascending=False
        )

    def numeric_overview(self) -> pd.DataFrame:
        """Returns standard descriptive statistics for numeric columns only."""
        return self.df.describe().T

    def categorical_overview(self) -> pd.DataFrame:
        """Returns descriptive statistics for non-numeric columns only."""
        return self.df.select_dtypes(exclude=["number"]).describe().T

    def alerts(
        self,
        correlation_threshold: float = 0.8,
        high_cardinality_threshold: int = 50,
        imbalance_threshold: float = 0.8,
        skewness_threshold: float = 1.0,
        uniform_pvalue_threshold: float = 0.999,
        duplicate_threshold: int = 10,
    ) -> pd.DataFrame:
        """Calculates dataset and column-level quality alerts and returns a standard pandas DataFrame."""
        alerts_list: list[dict[str, str]] = []
        n_rows = len(self.df)

        # -------------------------------------------------------------------
        # 1. Dataset-Level Alerts
        # -------------------------------------------------------------------
        if self.df.empty:
            alerts_list.append(
                {"Description": "Dataset is empty", "Alert Type": "Empty"}
            )
            return pd.DataFrame(alerts_list)

        duplicate_count = self.df.duplicated().sum()

        if duplicate_count > duplicate_threshold:
            pct = round((duplicate_count / n_rows) * 100, 1)
            alerts_list.append(
                {
                    "Description": (
                        f"Dataset has {duplicate_count} ({pct}%) duplicate rows"
                    ),
                    "Alert Type": "Duplicates",
                }
            )

        # -------------------------------------------------------------------
        # 2. Correlation Alert (Dataset Level)
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # 3. Column-Level Alerts
        # -------------------------------------------------------------------
        for col in self.df.columns:
            s = self.df[col]
            clean_s = s.dropna()
            n_unique = clean_s.nunique()
            n_missing = s.isnull().sum()

            # Constant / Zeros
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

            # High Cardinality
            elif n_unique > high_cardinality_threshold:
                alerts_list.append(
                    {
                        "Description": (
                            f"{col} has a high cardinality: {n_unique} distinct values"
                        ),
                        "Alert Type": "High cardinality",
                    }
                )

            # Imbalance
            if not clean_s.empty and n_unique > 1:
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

            # Missing Values
            if n_missing > 0:
                pct = round((n_missing / n_rows) * 100, 1)
                alerts_list.append(
                    {
                        "Description": f"{col} has {n_missing} ({pct}%) missing values",
                        "Alert Type": "Missing",
                    }
                )

            # Infinite Values
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                inf_count = np.isinf(s).sum()
                if inf_count > 0:
                    alerts_list.append(
                        {
                            "Description": f"{col} has {inf_count} infinite values",
                            "Alert Type": "Infinite",
                        }
                    )

            # Unique Values
            if n_unique == n_rows and n_rows > 0:
                alerts_list.append(
                    {
                        "Description": f"{col} has all unique values",
                        "Alert Type": "Unique",
                    }
                )

            # Skewness & Uniformity
            if (
                pd.api.types.is_numeric_dtype(s)
                and not pd.api.types.is_bool_dtype(s)
                and len(clean_s) > 3
            ):
                col_skew = abs(clean_s.skew())  # type: ignore
                if col_skew >= skewness_threshold:  # type: ignore
                    alerts_list.append(
                        {
                            "Description": (
                                f"{col} is highly skewed ({round(col_skew, 2)})"  # type: ignore
                            ),
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
                except Exception:  # noqa: BLE001, S110
                    pass

        if not alerts_list:
            return pd.DataFrame(columns=["Description", "Alert Type"])

        return pd.DataFrame(alerts_list)
