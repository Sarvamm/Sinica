"""Polars + Plotly port of ``ProfileReport``.

A small, data-profiling-style tool for profiling and summarising Polars
DataFrames inside Jupyter notebooks.

Requirements (polars >= 1.0 recommended)::

    pip install polars plotly numpy scipy tqdm wordcloud ipython nbformat

Usage::

    report = ProfileReport(df)          # polars DataFrame/LazyFrame or pandas DataFrame
    report                              # (last line of a cell) renders the overview
    report.get_alerts()                 # alerts as a polars DataFrame
    report.univ("price", "numeric")     # per-column deep dive

Missing-value semantics
-----------------------
Polars distinguishes ``null`` from ``NaN``; pandas does not. To keep the
behaviour of the original report, *both* are treated as missing for float
columns.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np
import plotly.graph_objects as go
import polars as pl
import polars.selectors as cs
from IPython.display import Markdown, display
from plotly.subplots import make_subplots
from scipy import stats
from tqdm import tqdm
from wordcloud import WordCloud

try:  # pandas is optional; only needed to accept pandas DataFrames
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


# --------------------------------------------------------------------------- #
# Small module-level helpers
# --------------------------------------------------------------------------- #
def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KiB"
    return f"{n / 1024**2:.1f} MiB"


def _num(x) -> float:
    """Polars returns ``None`` for undefined stats; map that to NaN."""
    return float("nan") if x is None else float(x)


def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _nan_to_none(x):
    return None if _is_nan(x) else x


def _fmt(x) -> str:
    return "-" if _is_nan(x) else str(x)


def _pct(part: float, whole: float) -> float:
    return (part / whole) * 100 if whole > 0 else 0.0


class ProfileReport:
    """A tool for profiling and generating statistical summaries of Polars DataFrames."""

    # Max points sent to the browser for box plots (histograms are binned in
    # numpy, so they always use all of the data).
    MAX_PLOT_POINTS: int = 50_000
    PLOT_TEMPLATE: str = "plotly_white"

    @staticmethod
    def _from_pandas(pdf: "pd.DataFrame") -> pl.DataFrame:
        """Convert a pandas DataFrame to polars.

        The pandas index is dropped (reset it first if it carries information).
        If the direct conversion fails -- typically because an ``object``
        column mixes types -- those columns are cast to strings (keeping
        missing values as nulls) and the conversion is retried.
        """
        try:
            return pl.from_pandas(pdf, nan_to_null=True)
        except Exception:
            fixed = pdf.copy()
            for c in fixed.select_dtypes(include="object").columns:
                col = fixed[c]
                fixed[c] = col.where(col.isna(), col.astype(str))
            try:
                return pl.from_pandas(fixed, nan_to_null=True)
            except Exception as e:
                raise TypeError(
                    f"Could not convert the pandas DataFrame to polars: {e}"
                ) from e

    def __init__(self, df: pl.DataFrame | pl.LazyFrame | pd.DataFrame):
        if pd is not None and isinstance(df, pd.DataFrame):
            df = self._from_pandas(df)
        elif isinstance(df, pl.LazyFrame):
            df = df.collect()
        if not isinstance(df, pl.DataFrame):
            raise TypeError(
                "Input must be a polars DataFrame/LazyFrame or a pandas DataFrame."
            )

        self.df = df

        # Core attributes
        self.num_vars: int = 0
        self.num_obs: int = 0
        self.total_cells: int = 0
        self.missing_cells_count: int = 0
        self.missing_pct: float = 0.0
        self.dup_rows: int = 0
        self.dup_pct: float = 0.0

        # Analysis outputs
        self.basic_stats_df: pl.DataFrame = pl.DataFrame()
        self.dtypes: dict[str, str] = {}
        self.dtype_counts_df: pl.DataFrame = pl.DataFrame()
        self.summary_df: pl.DataFrame = pl.DataFrame()

        self.missing_cols: list[str] = []
        self.missing_df: pl.DataFrame = pl.DataFrame()
        self.missing_corr: pl.DataFrame = pl.DataFrame()

        # Internal caches
        self._missing_mask: pl.DataFrame = pl.DataFrame()
        self._missing_counts: dict[str, int] = {}
        self._corr_cache: tuple[list[str], np.ndarray] | None = None

        self._profile_dataframe()

    # ------------------------------------------------------------------ #
    # Generic helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_hashable_dtype(dtype: pl.DataType) -> bool:
        """Object and nested dtypes (List/Array/Struct) are skipped for
        duplicate / frequency checks, mirroring the unhashable check in the
        pandas version."""
        return dtype != pl.Object and not dtype.is_nested()

    def _get_hashable_subset(self) -> list[str]:
        """Columns that can be safely used for duplicate checks."""
        return [c for c, dt in self.df.schema.items() if self._is_hashable_dtype(dt)]

    @staticmethod
    def _clean(s: pl.Series) -> pl.Series:
        """Drop nulls (and NaN for float columns)."""
        out = s.drop_nulls()
        if s.dtype.is_float():
            out = out.drop_nans()
        return out

    @staticmethod
    def _missing_count(s: pl.Series) -> int:
        n = s.null_count()
        if s.dtype.is_float():
            n += int(s.is_nan().sum() or 0)
        return int(n)

    @staticmethod
    def _n_unique(s: pl.Series) -> int | None:
        try:
            return int(s.n_unique())
        except Exception:
            return None

    @staticmethod
    def _top_frequency(clean: pl.Series) -> float | None:
        try:
            vc = clean.alias("value").value_counts(sort=True)
            return float(vc[0, vc.width - 1]) / clean.len()
        except Exception:
            return None

    @staticmethod
    def _top_value(clean: pl.Series) -> tuple[str | None, int | None]:
        if clean.len() == 0:
            return None, None
        try:
            vc = clean.alias("value").value_counts(sort=True)
            return str(vc[0, 0]), int(vc[0, vc.width - 1])
        except Exception:
            return None, None

    @staticmethod
    def _kv_table(
        key_name: str, items: list[tuple[str, object]], value_name: str = "Value"
    ) -> pl.DataFrame:
        """Two-column string table (mixed-type values are stringified, since a
        polars column has a single dtype)."""
        return pl.DataFrame(
            {
                key_name: [k for k, _ in items],
                value_name: [str(v) for _, v in items],
            }
        )

    @staticmethod
    def _show_table(df: pl.DataFrame) -> None:
        """Display a polars DataFrame without row/column truncation."""
        with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=100):
            display(df)

    def _show_fig(self, fig: go.Figure, height: int | None = None) -> None:
        fig.update_layout(
            template=self.PLOT_TEMPLATE, margin=dict(l=60, r=30, t=70, b=50)
        )
        if height:
            fig.update_layout(height=height)
        fig.show()

    def _plot_sample(self, arr: np.ndarray) -> np.ndarray:
        if arr.size <= self.MAX_PLOT_POINTS:
            return arr
        rng = np.random.default_rng(0)
        return rng.choice(arr, self.MAX_PLOT_POINTS, replace=False)

    def _check_col(self, col: str) -> None:
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    @staticmethod
    def _pairwise_corr(frame: pl.DataFrame) -> np.ndarray:
        """Pairwise-complete Pearson correlation matrix (like ``DataFrame.corr``
        in pandas), computed with native polars expressions."""
        cols = frame.columns
        k = len(cols)
        mat = np.full((k, k), np.nan)
        if k == 0:
            return mat

        frame = frame.cast(pl.Float64).fill_nan(None)
        exprs: list[pl.Expr] = []
        pairs: list[tuple[int, int]] = []
        for i in range(k):
            for j in range(i, k):
                a, b = pl.col(cols[i]), pl.col(cols[j])
                valid = a.is_not_null() & b.is_not_null()
                exprs.append(
                    pl.corr(a.filter(valid), b.filter(valid)).alias(f"{i}_{j}")
                )
                pairs.append((i, j))

        values = frame.select(exprs).row(0)
        for (i, j), v in zip(pairs, values):
            mat[i, j] = mat[j, i] = np.nan if v is None else v
        return mat

    def _numeric_corr(self) -> tuple[list[str], np.ndarray]:
        """Cached (labels, correlation matrix) for numeric columns."""
        if self._corr_cache is None:
            num = self.df.select(cs.numeric())
            labels = num.columns
            mat = self._pairwise_corr(num) if len(labels) > 1 else np.empty((0, 0))
            self._corr_cache = (labels, mat)
        return self._corr_cache

    # ------------------------------------------------------------------ #
    # Profiling pipeline
    # ------------------------------------------------------------------ #
    def _profile_dataframe(self) -> None:
        """Executes initialization steps sequentially with progress bar feedback."""
        with tqdm(total=4, desc="Understanding DataFrame") as pbar:
            pbar.set_description("Calculating dataset dimensions & missing statistics")
            self._compute_basic_stats()
            pbar.update(1)

            pbar.set_description("Analyzing data types")
            self._compute_dtype_counts()
            pbar.update(1)

            pbar.set_description("Building statistical summary table")
            self._compute_summary()
            pbar.update(1)

            pbar.set_description("Analyzing missing value distributions")
            self._compute_missing_analysis()
            pbar.update(1)

    def _build_missing_mask(self) -> pl.DataFrame:
        """Boolean frame, True where a cell is null (or NaN for floats)."""
        exprs = []
        for name, dtype in self.df.schema.items():
            e = pl.col(name).is_null()
            if dtype.is_float():
                e = e | pl.col(name).is_nan()
            exprs.append(e.alias(name))
        return self.df.select(exprs) if exprs else pl.DataFrame()

    def _compute_basic_stats(self) -> None:
        self.num_vars = self.df.width
        self.num_obs = self.df.height
        self.total_cells = self.num_vars * self.num_obs

        self._missing_mask = self._build_missing_mask()
        if self._missing_mask.width:
            row = self._missing_mask.select(pl.all().sum()).row(0, named=True)
            self._missing_counts = {k: int(v or 0) for k, v in row.items()}
        else:
            self._missing_counts = {}

        self.missing_cells_count = sum(self._missing_counts.values())
        self.missing_pct = _pct(self.missing_cells_count, self.total_cells)

        # Duplicates (rows identical to an earlier row, as in pandas `duplicated`)
        hashable_cols = self._get_hashable_subset()
        if hashable_cols and self.num_obs > 0:
            self.dup_rows = self.num_obs - int(self.df.select(hashable_cols).n_unique())
        else:
            self.dup_rows = 0
        self.dup_pct = _pct(self.dup_rows, self.num_obs)

        self.basic_stats_df = self._kv_table(
            "Metric",
            [
                ("Number of variables", self.num_vars),
                ("Number of observations", self.num_obs),
                ("Missing cells", self.missing_cells_count),
                ("Missing cells (%)", f"{self.missing_pct:.1f}%"),
                ("Duplicate rows", self.dup_rows),
                ("Duplicate rows (%)", f"{self.dup_pct:.1f}%"),
            ],
        )

    def _compute_dtype_counts(self) -> None:
        self.dtypes = {c: str(dt) for c, dt in self.df.schema.items()}
        counts = Counter(self.dtypes.values())
        self.dtype_counts_df = pl.DataFrame(
            {"Data Type": list(counts.keys()), "Count": list(counts.values())},
            schema={"Data Type": pl.String, "Count": pl.Int64},
        ).sort("Count", descending=True)

    def _compute_summary(self) -> None:
        """One row per variable (the pandas version returned the transpose)."""
        schema = {
            "column": pl.String,
            "dtype": pl.String,
            "count": pl.Int64,
            "unique": pl.Int64,
            "missing%": pl.Float64,
            "mean": pl.Float64,
            "std": pl.Float64,
            "min": pl.Float64,
            "median": pl.Float64,
            "max": pl.Float64,
            "top": pl.String,
            "freq": pl.Int64,
        }

        rows: list[dict] = []
        for col in self.df.columns:
            s = self.df[col]
            clean = self._clean(s)
            n_missing = self._missing_counts.get(col, 0)
            n_unique = self._n_unique(clean)

            row = {k: None for k in schema}
            row.update(
                column=col,
                dtype=str(s.dtype),
                count=clean.len(),
                # like nunique(dropna=False): missing counts as one more value
                unique=None if n_unique is None else n_unique + (1 if n_missing else 0),
                **{"missing%": round(_pct(n_missing, self.num_obs), 2)},
            )

            if s.dtype.is_numeric():
                if clean.len() > 0:
                    f = clean.cast(pl.Float64)
                    row.update(
                        mean=_nan_to_none(_num(f.mean())),
                        std=_nan_to_none(_num(f.std())),
                        min=_nan_to_none(_num(f.min())),
                        median=_nan_to_none(_num(f.median())),
                        max=_nan_to_none(_num(f.max())),
                    )
            else:
                row["top"], row["freq"] = self._top_value(clean)

            rows.append(row)

        self.summary_df = pl.DataFrame(rows, schema=schema).sort(
            "dtype", maintain_order=True
        )

    def _compute_missing_analysis(self) -> None:
        self.missing_cols = [c for c, n in self._missing_counts.items() if n > 0]

        names = list(self._missing_counts.keys())
        counts = [self._missing_counts[c] for c in names]
        self.missing_df = pl.DataFrame(
            {
                "column": names,
                "missing_count": counts,
                "missing_pct": [round(_pct(n, self.num_obs), 2) for n in counts],
            },
            schema={
                "column": pl.String,
                "missing_count": pl.Int64,
                "missing_pct": pl.Float64,
            },
        ).sort("missing_count", descending=True)

        if len(self.missing_cols) > 1:
            indicators = self._missing_mask.select(self.missing_cols)
            mat = self._pairwise_corr(indicators)
            corr = pl.DataFrame({c: mat[:, i] for i, c in enumerate(self.missing_cols)})
            self.missing_corr = pl.concat(
                [pl.DataFrame({"column": self.missing_cols}), corr],
                how="horizontal",
            )
        else:
            self.missing_corr = pl.DataFrame()

    # ------------------------------------------------------------------ #
    # Overview
    # ------------------------------------------------------------------ #
    def show_overview(self) -> None:
        """Renders the complete data overview directly in Jupyter Notebooks."""
        display(Markdown("# Data Overview"))
        display(Markdown("## Basic stats:"))
        self._show_table(self.get_basic_stats())
        display(Markdown("## Variable types:"))
        self._show_table(self.get_dtype_counts())
        display(Markdown("## Data Alerts"))
        self._show_table(self.get_alerts())
        display(Markdown("## Data Summary:"))
        self._show_table(self.get_summary())
        self.show_missing_report()

        labels, corr = self._numeric_corr()
        if len(labels) > 1:
            self._corr_heatmap(labels, corr, "Correlation heatmap")

    def _ipython_display_(self) -> None:
        self.show_overview()

    def get_basic_stats(self) -> pl.DataFrame:
        return self.basic_stats_df

    def get_dtype_counts(self) -> pl.DataFrame:
        return self.dtype_counts_df

    def get_summary(self) -> pl.DataFrame:
        return self.summary_df

    def _corr_heatmap(
        self,
        labels: list[str],
        matrix: np.ndarray,
        title: str,
        mask_upper: bool = False,
    ) -> None:
        z = np.array(matrix, dtype=float)
        if mask_upper:  # hide upper triangle + diagonal, like the seaborn mask
            z[np.triu_indices_from(z)] = np.nan

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                zmin=-1,
                zmax=1,
                colorscale="RdBu_r",
                texttemplate="%{z:.2f}",
                hoverongaps=False,
                colorbar=dict(title="r"),
            )
        )
        size = int(min(900, max(450, 60 * len(labels) + 150)))
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category", autorange="reversed")
        fig.update_layout(title=title, width=size + 100, height=size)
        self._show_fig(fig)

    def show_missing_report(self, max_matrix_rows: int = 5_000) -> None:
        """Missing-value tables and charts.

        ``max_matrix_rows`` caps the number of rows drawn in the missing-value
        matrix (rows are thinned evenly) so large frames stay responsive.
        """
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
        self._show_table(self.missing_df)

        # 1. Percentage of missing values by variable (only affected columns)
        miss = self.missing_df.filter(pl.col("missing_count") > 0).sort("missing_pct")
        pcts = miss["missing_pct"].to_list()
        fig = go.Figure(
            go.Bar(
                x=pcts,
                y=miss["column"].to_list(),
                orientation="h",
                text=[f"{p:.1f}%" for p in pcts],
                textposition="auto",
                marker_color="#4c72b0",
            )
        )
        fig.update_yaxes(type="category")
        fig.update_layout(
            title="Percentage of Missing Values by variable",
            xaxis_title="Percentage of Missing Values",
        )
        self._show_fig(fig, height=max(300, 28 * len(pcts) + 140))

        # 2. Missing value matrix
        step = max(1, math.ceil(self.num_obs / max_matrix_rows))
        sampled = self._missing_mask.gather_every(step)
        z = sampled.to_numpy().T.astype(np.int8)
        title = "Missing Value matrix"
        if step > 1:
            title += f" (every {step}th row shown)"
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=np.arange(0, self.num_obs, step),
                y=sampled.columns,
                colorscale=[[0, "#f2f2f2"], [1, "#222222"]],
                showscale=False,
                xgap=0,
                ygap=1,
                hovertemplate="row %{x}<br>%{y}<br>missing=%{z}<extra></extra>",
            )
        )
        fig.update_yaxes(type="category", autorange="reversed")
        fig.update_layout(title=title, xaxis_title="Row index")
        self._show_fig(fig, height=max(300, 22 * self.num_vars + 140))

        # 3. Missing value correlation (lower triangle)
        if not self.missing_corr.is_empty() and len(self.missing_cols) > 1:
            mat = self.missing_corr.drop("column").to_numpy()
            self._corr_heatmap(
                self.missing_cols, mat, "Missing Value Correlation", mask_upper=True
            )

    # ------------------------------------------------------------------ #
    # Alerts
    # ------------------------------------------------------------------ #
    def get_alerts(
        self,
        correlation_threshold: float = 0.8,
        high_cardinality_threshold: int = 50,
        imbalance_threshold: float = 0.8,
        skewness_threshold: float = 1.0,
        uniform_pvalue_threshold: float = 0.999,
        duplicate_threshold: int = 10,
    ) -> pl.DataFrame:
        alerts: list[dict[str, str]] = []
        schema = {"Description": pl.String, "Alert Type": pl.String}

        def add(description: str, alert_type: str) -> None:
            alerts.append({"Description": description, "Alert Type": alert_type})

        if self.df.is_empty():
            return pl.DataFrame(
                [{"Description": "Dataset is empty", "Alert Type": "Empty"}],
                schema=schema,
            )

        if self.dup_rows > duplicate_threshold:
            add(
                f"Dataset has {self.dup_rows} ({self.dup_pct:.1f}%) duplicate rows",
                "Duplicates",
            )

        # High correlation
        labels, corr = self._numeric_corr()
        if len(labels) > 1:
            abs_corr = np.abs(corr)
            for i, col in enumerate(labels):
                hits = [
                    labels[j]
                    for j in range(len(labels))
                    if j != i and abs_corr[i, j] >= correlation_threshold
                ]
                if hits:
                    other = len(hits) - 1
                    other_text = f" and {other} other fields" if other > 0 else ""
                    add(
                        f"{col} is highly correlated with {hits[0]}{other_text}",
                        "High correlation",
                    )

        for col in self.df.columns:
            s = self.df[col]
            clean = self._clean(s)
            n_unique = self._n_unique(clean)
            n_missing = self._missing_counts.get(col, 0)
            hashable = self._is_hashable_dtype(s.dtype)

            if n_unique == 1:
                val = clean[0]
                add(f'{col} has constant value "{val}"', "Constant")
                if val == 0 and not isinstance(val, bool):
                    add(f"{col} contains only zeros", "Zeros")
            elif n_unique is not None and n_unique > high_cardinality_threshold:
                add(
                    f"{col} has a high cardinality: {n_unique} distinct values",
                    "High cardinality",
                )

            if n_unique is not None and n_unique > 1 and hashable:
                top_freq = self._top_frequency(clean)
                if top_freq is not None and top_freq >= imbalance_threshold:
                    add(
                        f"{col} is highly imbalanced ({round(top_freq * 100, 1)}% top value)",
                        "Imbalance",
                    )

            if n_missing > 0:
                pct = round(_pct(n_missing, self.num_obs), 1)
                add(f"{col} has {n_missing} ({pct}%) missing values", "Missing")

            if s.dtype.is_float():
                inf_count = int(s.is_infinite().sum() or 0)
                if inf_count > 0:
                    add(f"{col} has {inf_count} infinite values", "Infinite")

            if n_unique == self.num_obs and self.num_obs > 0 and hashable:
                add(f"{col} has all unique values", "Unique")

            if s.dtype.is_numeric() and clean.len() > 3:
                finite = clean.cast(pl.Float64)
                finite = finite.filter(finite.is_finite())
                if finite.len() > 3:
                    col_skew = abs(_num(finite.skew(bias=False)))
                    if not math.isnan(col_skew) and col_skew >= skewness_threshold:
                        add(
                            f"{col} is highly skewed ({round(col_skew, 2)})",
                            "Skewed",
                        )

                try:
                    counts, _ = np.histogram(clean.to_numpy(), bins=10)
                    _, p_val = stats.chisquare(counts)
                    if p_val >= uniform_pvalue_threshold:
                        add(f"{col} follows a uniform distribution", "Uniform")
                except Exception:
                    pass

        return pl.DataFrame(alerts, schema=schema)

    # ------------------------------------------------------------------ #
    # Univariate: text
    # ------------------------------------------------------------------ #
    def text_univ(self, col: str) -> None:
        """Text statistics (length, unicode, samples, word frequencies) plus a word cloud."""
        self._check_col(col)

        n = self.df.height
        s = self.df[col]
        clean = self._clean(s).cast(pl.String)

        # 1. Basic metrics
        distinct = clean.n_unique()
        distinct_pct = _pct(distinct, n)
        missing = self._missing_count(s)
        missing_pct = _pct(missing, n)
        mem_str = _format_bytes(s.estimated_size())

        # 2. Length statistics
        lengths = clean.str.len_chars()
        has_data = lengths.len() > 0
        max_len = lengths.max() if has_data else 0
        min_len = lengths.min() if has_data else 0
        mean_len = _num(lengths.mean()) if has_data else 0.0
        median_len = _num(lengths.median()) if has_data else 0.0

        # 3. Characters & unicode
        all_text = "".join(clean.to_list())
        total_chars = len(all_text)
        distinct_chars = len(set(all_text))
        distinct_categories = len({unicodedata.category(c) for c in all_text})
        # NOTE: scripts / blocks are placeholders (as in the original); real
        # values would need a Unicode database such as `unicodedata2`.
        distinct_scripts = 1 if all_text else 0
        distinct_blocks = 1 if all_text else 0

        # 4. Word frequencies. Rows are joined with a newline so the last word
        # of one row never fuses with the first word of the next.
        corpus = "\n".join(clean.to_list())
        words = re.findall(r"\b\w+\b", corpus.lower())
        word_counts = Counter(words)
        most_common = word_counts.most_common(5)
        bottom_common = sorted(word_counts.items(), key=lambda x: (x[1], x[0]))[:5]

        # 5. Sample rows
        sample_rows = clean.head(5).to_list()
        sample_rows += ["-"] * (5 - len(sample_rows))

        # --- Display ---
        display(Markdown(f"### **{col}**"))
        display(Markdown("*Text*"))

        display(Markdown("#### Overview"))
        self._show_table(
            self._kv_table(
                "Metric",
                [
                    ("Distinct", distinct),
                    ("Distinct (%)", f"{distinct_pct:.1f}%"),
                    ("Missing", missing),
                    ("Missing (%)", f"{missing_pct:.1f}%"),
                    ("Memory size", mem_str),
                ],
            )
        )

        display(Markdown("#### Length statistics"))
        self._show_table(
            self._kv_table(
                "Length Metric",
                [
                    ("Max length", max_len),
                    ("Median length", median_len),
                    ("Mean length", f"{mean_len:.4f}"),
                    ("Min length", min_len),
                ],
            )
        )

        display(Markdown("#### Characters and Unicode"))
        self._show_table(
            self._kv_table(
                "Characters and Unicode",
                [
                    ("Total characters", f"{total_chars:,}"),
                    ("Distinct characters", distinct_chars),
                    ("Distinct categories", distinct_categories),
                    ("Distinct scripts", distinct_scripts),
                    ("Distinct blocks", distinct_blocks),
                ],
            )
        )

        display(Markdown("#### Unique statistics"))
        self._show_table(
            self._kv_table(
                "Unique",
                [("Unique", distinct), ("Unique (%)", f"{distinct_pct:.1f}%")],
            )
        )

        display(Markdown("#### Sample rows"))
        self._show_table(
            self._kv_table(
                "Sample",
                list(
                    zip(
                        ["1st row", "2nd row", "3rd row", "4th row", "5th row"],
                        sample_rows,
                    )
                ),
            )
        )

        display(Markdown("#### Word Frequency Analysis"))
        top_words_df = pl.DataFrame(
            most_common,
            schema={"Most Frequent Word": pl.String, "Count": pl.Int64},
            orient="row",
        )
        bottom_words_df = pl.DataFrame(
            bottom_common,
            schema={"Least Frequent Word": pl.String, "Count": pl.Int64},
            orient="row",
        )
        display(Markdown("**Top 5 Most Frequent Words:**"))
        self._show_table(top_words_df)
        display(Markdown("**Bottom 5 Least Frequent Words:**"))
        self._show_table(bottom_words_df)

        # 6. Word cloud (rendered as an image trace)
        if word_counts:
            try:
                wc = WordCloud(
                    width=800,
                    height=400,
                    background_color="white",
                    colormap="Dark2",
                    max_words=150,
                    contour_width=1,
                ).generate(corpus)
            except ValueError:  # e.g. only single-character tokens
                return

            fig = go.Figure(go.Image(z=wc.to_array()))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(
                title=dict(text=f"<b>Word Cloud of {col}</b>"), width=820, height=470
            )
            self._show_fig(fig)

    # ------------------------------------------------------------------ #
    # Univariate: numeric
    # ------------------------------------------------------------------ #
    def numeric_univ(self, col: str) -> None:
        """Basic, quantile and descriptive statistics, plus histogram (with KDE) and boxplot."""
        self._check_col(col)

        s = self.df[col]
        if not s.dtype.is_numeric():  # excludes Boolean
            raise TypeError(f"Column '{col}' must be a continuous numeric type.")

        n = s.len()
        clean = self._clean(s)
        has_data = clean.len() > 0
        clean_f = clean.cast(pl.Float64)

        # --- 1. Basic metrics ---
        distinct = clean.n_unique()
        distinct_pct = _pct(distinct, n)
        missing = self._missing_count(s)
        missing_pct = _pct(missing, n)
        infinite = int(s.is_infinite().sum() or 0) if s.dtype.is_float() else 0
        infinite_pct = _pct(infinite, n)
        zeros = int((s == 0).sum() or 0)
        zeros_pct = _pct(zeros, n)
        negative = int((s < 0).sum() or 0)
        negative_pct = _pct(negative, n)

        mean_val = _num(clean_f.mean()) if has_data else np.nan
        min_val = clean.min() if has_data else None
        max_val = clean.max() if has_data else None
        mem_str = _format_bytes(s.estimated_size())

        # --- 2. Quantiles (linear interpolation, as pandas; polars defaults to "nearest") ---
        def q(p: float) -> float:
            return (
                _num(clean_f.quantile(p, interpolation="linear"))
                if has_data
                else np.nan
            )

        q1, median_val, q3 = q(0.25), q(0.5), q(0.75)
        p5, p95 = q(0.05), q(0.95)
        iqr = q3 - q1
        rng = _num(max_val) - _num(min_val) if has_data else np.nan

        # --- 3. Descriptive statistics (bias-corrected, matching pandas) ---
        std_val = _num(clean_f.std()) if has_data else np.nan
        cv_val = (
            std_val / abs(mean_val)
            if mean_val != 0 and not np.isnan(mean_val)
            else np.nan
        )
        kurt_val = (
            _num(clean_f.kurtosis(fisher=True, bias=False)) if has_data else np.nan
        )
        skew_val = _num(clean_f.skew(bias=False)) if has_data else np.nan
        sum_val = clean.sum() if has_data else np.nan
        var_val = _num(clean_f.var()) if has_data else np.nan
        mad_val = _num((clean_f - median_val).abs().median()) if has_data else np.nan

        # Monotonicity (like pandas: any missing value => not monotonic)
        if missing == 0:
            is_inc = s.is_sorted()
            is_dec = s.is_sorted(descending=True)
        else:
            is_inc = is_dec = False
        if is_inc and is_dec:
            monotonicity = "Constant"
        elif is_inc:
            monotonicity = "Monotonic increasing"
        elif is_dec:
            monotonicity = "Monotonic decreasing"
        else:
            monotonicity = "Not monotonic"

        # --- Display ---
        display(Markdown(f"### **{col}**"))
        display(Markdown(f"*Real number (ℝ)* — DataType: `{s.dtype}`"))

        display(Markdown("#### Overview statistics"))
        basic_table = pl.DataFrame(
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
                    str(distinct),
                    f"{distinct_pct:.1f}%",
                    str(missing),
                    f"{missing_pct:.1f}%",
                    str(infinite),
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
                    _fmt(min_val),
                    _fmt(max_val),
                    str(zeros),
                    f"{zeros_pct:.1f}%",
                    str(negative),
                    f"{negative_pct:.1f}%",
                    mem_str,
                ],
            }
        )
        self._show_table(basic_table)

        display(Markdown("#### Quantile & Descriptive statistics"))
        quantile_table = pl.DataFrame(
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
                "Value": [
                    _nan_to_none(None if min_val is None else float(min_val)),
                    _nan_to_none(p5),
                    _nan_to_none(q1),
                    _nan_to_none(median_val),
                    _nan_to_none(q3),
                    _nan_to_none(p95),
                    _nan_to_none(None if max_val is None else float(max_val)),
                    _nan_to_none(rng),
                    _nan_to_none(iqr),
                ],
            },
            schema={"Quantile statistic": pl.String, "Value": pl.Float64},
        )

        def sci_or_plain(x) -> str:
            if isinstance(x, float) and np.isnan(x):
                return "nan"
            return f"{x:.6e}" if abs(x) > 1e6 else f"{x}"

        desc_table = pl.DataFrame(
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
                    f"{cv_val:.8f}" if not np.isnan(cv_val) else "-",
                    f"{kurt_val:.7f}" if not np.isnan(kurt_val) else "-",
                    f"{mean_val:.3f}" if not np.isnan(mean_val) else "-",
                    f"{mad_val}" if not np.isnan(mad_val) else "-",
                    f"{skew_val:.8f}" if not np.isnan(skew_val) else "-",
                    sci_or_plain(sum_val),
                    sci_or_plain(var_val),
                    monotonicity,
                ],
            }
        )
        self._show_table(quantile_table)
        self._show_table(desc_table)

        # --- Histogram (+ KDE) and boxplot ---
        x = clean_f.filter(clean_f.is_finite()).to_numpy()
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"Histogram of {col}", f"Boxplot of {col}"),
        )

        if x.size:
            edges = np.histogram_bin_edges(x, bins="auto")
            n_bins = max(1, min(len(edges) - 1, 100))
            counts, edges = np.histogram(x, bins=n_bins)
            width = float(edges[1] - edges[0])
            centers = (edges[:-1] + edges[1:]) / 2

            fig.add_trace(
                go.Bar(
                    x=centers,
                    y=counts,
                    width=width,
                    marker_color="dodgerblue",
                    opacity=0.6,
                    showlegend=False,
                    hovertemplate="%{x}<br>count=%{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # KDE overlay, scaled to counts (like seaborn's kde=True)
            try:
                if np.unique(x[:10_000]).size > 1:
                    kde_sample = x
                    if x.size > 10_000:
                        kde_sample = np.random.default_rng(0).choice(
                            x, 10_000, replace=False
                        )
                    kde = stats.gaussian_kde(kde_sample)
                    grid = np.linspace(edges[0], edges[-1], 200)
                    fig.add_trace(
                        go.Scatter(
                            x=grid,
                            y=kde(grid) * x.size * width,
                            mode="lines",
                            line=dict(color="navy", width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=1,
                    )
            except Exception:
                pass

            fig.add_trace(
                go.Box(
                    x=self._plot_sample(x),
                    name="",
                    orientation="h",
                    marker_color="mediumseagreen",
                    boxpoints="outliers",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        fig.update_xaxes(title_text=col, row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text=col, row=1, col=2)
        fig.update_layout(bargap=0.02)
        self._show_fig(fig, height=420)

    # ------------------------------------------------------------------ #
    # Univariate: date
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_dates(s: pl.Series) -> pl.Series:
        """Parse a column into Date/Datetime, turning unparseable values into null."""
        if s.dtype == pl.Date or s.dtype == pl.Datetime:
            return s
        if s.dtype in (pl.String, pl.Categorical):
            s = s.cast(pl.String)
            best: pl.Series | None = None
            for parse in (
                lambda x: x.str.to_datetime(strict=False),
                lambda x: x.str.to_date(strict=False),
            ):
                try:
                    cand = parse(s)
                except Exception:
                    continue
                if best is None or cand.null_count() < best.null_count():
                    best = cand
            if best is not None:
                return best
            return pl.Series(s.name, [None] * s.len(), dtype=pl.Datetime)
        raise TypeError(
            f"Column '{s.name}' has dtype {s.dtype}; expected Date, Datetime or String."
        )

    def date_univ(self, col: str) -> None:
        """Date/datetime statistics and a time-series histogram."""
        self._check_col(col)

        n = self.df.height
        s = self.df[col]

        parsed = self._parse_dates(s)
        invalid_count = max(0, parsed.null_count() - s.null_count())
        invalid_pct = _pct(invalid_count, n)

        clean = parsed.drop_nulls()

        # --- Metrics ---
        distinct = clean.n_unique()
        distinct_pct = _pct(distinct, n)
        missing = s.null_count() + invalid_count
        missing_pct = _pct(missing, n)
        min_val = clean.min() if clean.len() else "-"
        max_val = clean.max() if clean.len() else "-"
        mem_str = _format_bytes(s.estimated_size())

        # --- Display ---
        display(Markdown(f"### **{col}**"))
        display(Markdown("*Date*"))

        if clean.len() > 0 and distinct == clean.len() and n > 0:
            display(
                Markdown(
                    "<span"
                    " style='background-color:#d9534f; color:white; padding:2px 8px;"
                    " border-radius:4px; font-weight:bold; font-size:12px;'>Unique</span>"
                )
            )

        display(Markdown("#### Overview statistics"))
        self._show_table(
            self._kv_table(
                "Metric (Left)",
                [
                    ("Distinct", distinct),
                    ("Distinct (%)", f"{distinct_pct:.1f}%"),
                    ("Missing", missing),
                    ("Missing (%)", f"{missing_pct:.1f}%"),
                    ("Memory size", mem_str),
                ],
                value_name="Value (Left)",
            )
        )
        self._show_table(
            self._kv_table(
                "Metric (Right)",
                [
                    ("Minimum", min_val),
                    ("Maximum", max_val),
                    ("Invalid dates", invalid_count),
                    ("Invalid dates (%)", f"{invalid_pct:.1f}%"),
                ],
                value_name="Value (Right)",
            )
        )

        # --- Histogram over time (binned in numpy; 50 bins) ---
        if clean.is_empty():
            print(f"Warning: Column '{col}' contains no valid dates to plot.")
            return

        ts = clean.cast(pl.Datetime("us")).dt.epoch("us").to_numpy()
        counts, edges = np.histogram(ts, bins=50)
        to_dt = lambda a: np.round(a).astype("int64").astype("datetime64[us]")  # noqa: E731
        centers = to_dt((edges[:-1] + edges[1:]) / 2)
        width_ms = float(edges[1] - edges[0]) / 1000.0  # date-axis bar width is in ms

        fig = go.Figure(
            go.Bar(
                x=centers,
                y=counts,
                width=width_ms,
                marker_color="dodgerblue",
                hovertemplate="%{x}<br>count=%{y}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text=f"<b>Histogram of {col}</b>"),
            xaxis=dict(title=col, type="date", tickangle=-45),
            yaxis_title="Frequency",
            bargap=0.02,
        )
        self._show_fig(fig, height=400)

    # ------------------------------------------------------------------ #
    # Univariate: categorical
    # ------------------------------------------------------------------ #
    def categorical_univ(self, col: str) -> None:
        """Overview + common values tables and a proportional stacked bar of the top 5 values."""
        self._check_col(col)

        n = self.df.height
        s = self.df[col]
        clean = self._clean(s)

        # --- Metrics ---
        distinct = clean.n_unique()
        distinct_pct = _pct(distinct, n)
        missing = self._missing_count(s)
        missing_pct = _pct(missing, n)
        mem_str = _format_bytes(s.estimated_size())

        total_valid = clean.len()
        vc = clean.alias("value").value_counts(sort=True)
        values = vc.get_column("value")
        cnts = vc.get_column(vc.columns[-1])

        # --- Display ---
        display(Markdown(f"### **{col}**"))
        display(Markdown("*Categorical*"))

        display(Markdown("#### Overview statistics"))
        self._show_table(
            self._kv_table(
                "Metric",
                [
                    ("Distinct", distinct),
                    ("Distinct (%)", f"{distinct_pct:.1f}%"),
                    ("Missing", missing),
                    ("Missing (%)", f"{missing_pct:.1f}%"),
                    ("Memory size", mem_str),
                ],
            )
        )

        # --- Segmented "Common Values (Plot)" bar (top 5 categories) ---
        if total_valid > 0:
            colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
            fig = go.Figure()
            for i, (cat, count) in enumerate(
                zip(values.head(5).to_list(), cnts.head(5).to_list())
            ):
                pct = count / total_valid * 100
                fig.add_trace(
                    go.Bar(
                        x=[count],
                        y=[""],
                        orientation="h",
                        name=str(cat),
                        text=[f"{pct:.1f}%<br>({count})"],
                        textposition="inside",
                        insidetextanchor="middle",
                        textfont=dict(color="white", size=13),
                        marker=dict(
                            color=colors[i % len(colors)],
                            line=dict(color="white", width=1.5),
                        ),
                        hovertemplate=f"{cat}<br>{count} ({pct:.1f}%)<extra></extra>",
                    )
                )
            fig.update_layout(
                barmode="stack",
                title=dict(text="<b>Common Values (Plot)</b>", x=0.0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                legend=dict(orientation="h", yanchor="top", y=-0.05, x=0.0),
            )
            self._show_fig(fig, height=240)

        # --- Common values table ---
        display(Markdown("#### Common Values"))
        common_values_df = pl.DataFrame(
            {
                "Value": values,
                "Count": cnts,
                "Frequency (%)": [
                    f"{c / total_valid * 100:.1f}%" for c in cnts.to_list()
                ],
            }
        )
        self._show_table(common_values_df.head(10))

    # ------------------------------------------------------------------ #
    # Router
    # ------------------------------------------------------------------ #
    def univ(self, col: str, col_type: str) -> None:
        """Univariate analysis router requiring an explicit ``col_type``.

        Supported col_types: 'numeric', 'text', 'date', 'categorical' (with
        common aliases).
        """
        if col not in self.df.columns:
            raise ValueError(
                f"Column '{col}' does not exist in the provided DataFrame."
            )

        if col_type is None:
            raise ValueError(
                f"Missing required argument 'col_type' for column '{col}'. Please "
                "explicitly specify one of: 'numeric', 'text', 'date', 'categorical'."
            )

        col_type_norm = str(col_type).lower().strip()

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
            raise
