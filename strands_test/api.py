import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

# ── Load data once at startup ──────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "Consumer_Account_Tag.xlsx"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Excel file not found: {DATA_PATH}")

df = pd.read_excel(DATA_PATH)
# Normalise column names to lowercase with underscores (already are)
df.columns = [c.strip().lower() for c in df.columns]

app = FastAPI(
    title="Consumer Account Tag API",
    description="REST API for the Consumer_Account_Tag dataset.",
    version="1.0.0",
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "endpoints": ["/records", "/tags", "/accounts/{account_id}", "/schema"],
    }


@app.get("/schema")
def schema():
    """Return column names and data types."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


@app.get("/records")
def get_records(
    tag_nm: Optional[str] = Query(None, description="Filter by tag name (partial, case-insensitive)"),
    account_id: Optional[int] = Query(None, description="Filter by account_id"),
    accountnumber: Optional[str] = Query(None, description="Filter by account number (exact)"),
    limit: int = Query(100, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Row offset for pagination"),
):
    """Return records with optional filters and pagination."""
    result = df.copy()

    if tag_nm:
        result = result[result["tag_nm"].str.contains(tag_nm, case=False, na=False)]
    if account_id is not None:
        result = result[result["account_id"] == account_id]
    if accountnumber:
        result = result[result["accountnumber"] == accountnumber]

    total = len(result)
    page = result.iloc[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "data": page.to_dict(orient="records"),
    }


@app.get("/tags")
def get_tags():
    """Return all unique tag names and their counts."""
    counts = df["tag_nm"].value_counts().reset_index()
    counts.columns = ["tag_nm", "count"]
    return {"tags": counts.to_dict(orient="records")}


@app.get("/accounts/{account_id}")
def get_account(account_id: int):
    """Return all tag records for a specific account_id."""
    result = df[df["account_id"] == account_id]
    if result.empty:
        raise HTTPException(status_code=404, detail=f"account_id {account_id} not found")
    return {"account_id": account_id, "records": result.to_dict(orient="records")}


@app.get("/accounts")
def list_accounts():
    """Return all unique customer account numbers (deduplicated by accountnumber)."""
    result = (
        df[["accountnumber"]]
        .drop_duplicates()
        .sort_values("accountnumber")
        .to_dict(orient="records")
    )
    return {"total_unique_accounts": len(result), "accounts": result}
