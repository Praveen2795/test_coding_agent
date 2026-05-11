import os

import requests
from strands import Agent, tool
from strands.models.gemini import GeminiModel
from strands_tools import python_repl

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


@tool
def get_schema() -> str:
    """Fetch the current dataset schema: column names and their data types."""
    try:
        schema = requests.get(f"{API_BASE}/schema", timeout=5).json()
        return "Dataset columns:\n" + "\n".join(
            f"  - {col}: {dtype}" for col, dtype in schema.items()
        )
    except Exception as e:
        return f"Error fetching schema: {e}"


@tool
def get_sample_rows() -> str:
    """Fetch 3 sample rows to understand real data values and formats."""
    try:
        data = requests.get(f"{API_BASE}/records", params={"limit": 3}, timeout=5).json().get("data", [])
        return "Sample rows:\n" + "\n".join(str(r) for r in data)
    except Exception as e:
        return f"Error fetching sample rows: {e}"


def create_coder_agent(model: GeminiModel) -> Agent:
    return Agent(
        name="coder_agent",
        description=(
            "Writes and executes Python code for data analysis tasks. "
            "Accepts a plain-English task instruction and returns the JSON result "
            "of running the code — not the code itself. "
            "Handles its own write-run-fix loop internally."
        ),
        model=model,
        tools=[get_schema, get_sample_rows, python_repl],
        system_prompt=(
            "You are an expert Python data analyst who writes AND runs code.\n\n"
            "## Your job\n"
            "Given a plain-English task, produce the final JSON result by:\n"
            "  1. Understanding the data (via tools)\n"
            "  2. Writing the code\n"
            "  3. Running it with `python_repl`\n"
            "  4. Fixing and re-running if it errors (max 3 attempts)\n"
            "  5. Returning ONLY the raw JSON output from the successful run as your final response.\n\n"
            "## Workflow\n"
            "  1. Call `get_schema` to discover column names and data types.\n"
            "  2. Call `get_sample_rows` to see real data values and formats.\n"
            "  3. Write the complete Python code.\n"
            "  4. Call `python_repl` with the code.\n"
            "  5. If it errors: fix the code and call `python_repl` again. Repeat up to 3 times.\n"
            "  6. Once `python_repl` succeeds, copy the exact JSON line it printed and return it "
            "as your FINAL text response — nothing else, no explanation, just the raw JSON string.\n\n"
            "## Data source\n"
            f"Base URL: {API_BASE}\n"
            "- GET /records → params: limit (int), offset (int). Response key: 'data'\n"
            "- GET /schema  → column names and types\n\n"
            "## Code rules\n"
            "- Fetch ALL rows: "
            f"pd.DataFrame(requests.get('{API_BASE}/records', params={{'limit': 10000}}).json()['data'])\n"
            "- Do all filtering and aggregation in Python/pandas after fetching.\n"
            "- Read-only API — only GET requests. Never POST, PUT, PATCH, or DELETE.\n"
            "- Include all imports. Code must run as-is.\n\n"
            "## Date handling\n"
            "- Dates in this dataset are stored as short strings like '4/30/20' (m/d/yy, two-digit year).\n"
            "- Always parse date columns with: pd.to_datetime(df['col'], errors='coerce')\n"
            "- Never use a hardcoded format string like format='%m/%d/%Y'.\n\n"
            "## Column semantics\n"
            "- `account_id` is a row-level surrogate key — unique per row, meaningless for counting customers.\n"
            "- For unique account/customer counts always use `accountnumber.nunique()`.\n"
            "- For case-insensitive string filtering use: df[df['col'].str.strip().str.lower() == value.strip().lower()]\n\n"
            "## Output contract (the code you run must follow this)\n"
            "The last line of the code must print exactly one JSON object:\n"
            "  import json; print(json.dumps(result))\n"
            "where result = {'summary': '<one-line description>', 'data': <number|list|dict>}\n"
            "Print nothing else."
        ),
    )
