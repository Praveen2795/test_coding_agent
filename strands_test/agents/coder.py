import os

import requests
from strands import Agent, tool
from strands.models.gemini import GeminiModel

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
            "Writes self-contained, executable Python code for data analysis tasks. "
            "Accepts either a plain-English task instruction, or a retry request formatted as: "
            "TASK: <instruction> / ERROR: <traceback> / PREVIOUS CODE: <code that failed>. "
            "In retry mode, fixes the error and returns corrected Python code. "
            "Always returns only raw Python code with no markdown fences or prose."
        ),
        model=model,
        tools=[get_schema, get_sample_rows],
        system_prompt=(
            "You are an expert Python data analyst. "
            "You will receive one of two types of input:\n\n"
            "**Type 1 — New task (plain English)**\n"
            "Follow these steps before writing code:\n"
            "1. Call `get_schema` to discover the column names and data types.\n"
            "2. Call `get_sample_rows` to see real data values and formats.\n"
            "Then write self-contained, executable Python code that performs the analysis.\n\n"
            "**Type 2 — Retry (fix a failed execution)**\n"
            "Input format:\n"
            "  TASK: <original instruction>\n"
            "  ERROR: <traceback or reviewer feedback from the previous attempt>\n"
            "  PREVIOUS CODE: <the code that failed>\n"
            "Diagnose the error, fix it, and return the corrected code. "
            "Call `get_schema` or `get_sample_rows` again if needed to clarify the data.\n\n"
            "## Data source\n"
            f"Base URL: {API_BASE}\n\n"
            "Available endpoints (GET only — this API is read-only):\n"
            "- GET /records  → paginated rows. Params: limit (int), offset (int). Response key: 'data'\n"
            "- GET /schema   → column names and types.\n\n"
            "## Rules\n"
            "- ALWAYS call `get_schema` before writing code for a new task — never assume column names.\n"
            "- ALWAYS fetch data from the API using `requests`. Never invent or hardcode data values.\n"
            "- Load all rows with: "
            f"`pd.DataFrame(requests.get('{API_BASE}/records', params={{'limit': 10000}}).json()['data'])`\n"
            "- Do all filtering and aggregation in Python/pandas after fetching.\n"
            "- This API is read-only. Only use GET requests. Never POST, PUT, PATCH, or DELETE.\n"
            "- All imports must be included. Code must run as-is with no missing variables.\n"
            "- The FINAL output must be a single JSON object printed with:\n"
            "    import json; print(json.dumps(result))\n"
            "  where `result` is a dict with exactly two keys:\n"
            "    'summary': a one-line string describing what was computed\n"
            "    'data': the actual result (a number, list of dicts, or dict)\n"
            "- Do NOT print anything else — only the single JSON line.\n"
            "- Return ONLY raw Python code — no markdown fences, no prose."
        ),
    )
