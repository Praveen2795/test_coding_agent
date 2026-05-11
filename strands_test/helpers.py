"""
Shared helper utilities for generated data analysis code.

The coder agent imports these functions instead of writing its own
implementations. This ensures consistent, tested behaviour for
common operations regardless of how the LLM phrases the code.
"""

import pandas as pd


def parse_date(series: pd.Series) -> pd.Series:
    """
    Parse a Series of date strings into datetime, with no hardcoded format.
    Handles short years (4/30/20), long years (04/30/2020), ISO strings, etc.
    NaN/unparseable values become NaT instead of raising an error.
    """
    return pd.to_datetime(series, errors="coerce")


def fuzzy_match(series: pd.Series, value: str) -> pd.Series:
    """
    Case-insensitive, whitespace-stripped string comparison.
    Returns a boolean Series — use it as a filter mask.

    Example:
        df[fuzzy_match(df['tag_nm'], 'DM 809 IHPP Sent')]
    """
    return series.str.strip().str.lower() == value.strip().lower()


def parse_dollar(series: pd.Series) -> pd.Series:
    """
    Convert dollar-formatted strings like '$1,234.56' or '1234.56' to float.
    Non-parseable values become NaN.
    """
    return (
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )
