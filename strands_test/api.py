import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query

# ── Load data once at startup ──────────────────────────────────────────────────
DATA_FILE = os.getenv("DATA_FILE", "Consumer_Account_Tag.xlsx")
DATA_PATH = Path(__file__).parent.parent / DATA_FILE

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

if DATA_PATH.suffix.lower() in {".xlsx", ".xls"}:
    df = pd.read_excel(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)

df.columns = [c.strip().lower() for c in df.columns]

app = FastAPI(
    title=f"{DATA_FILE} API",
    description=f"Read-only REST API for {DATA_FILE}.",
    version="1.0.0",
)


# ── Read-only routes (GET only) ────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "endpoints": ["/records", "/schema"],
    }


@app.get("/schema")
def schema():
    """Return column names and data types."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


@app.get("/records")
def get_records(
    limit: int = Query(100, ge=1, le=10000, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Row offset for pagination"),
):
    """Return rows with pagination. Fetch all rows by setting limit=10000."""
    total = len(df)
    page = df.iloc[offset: offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "data": page.to_dict(orient="records"),
    }
