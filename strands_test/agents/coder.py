import requests
from strands import Agent
from strands.models.gemini import GeminiModel

API_BASE = "http://127.0.0.1:8000"


def _fetch_schema() -> str:
    """Fetch live schema from the API and format it as a readable string."""
    try:
        schema = requests.get(f"{API_BASE}/schema", timeout=5).json()
        lines = [f"  - {col}: {dtype}" for col, dtype in schema.items()]
        return "\n".join(lines)
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach the data API at {API_BASE}. "
            "Make sure the FastAPI server is running before starting main.py. "
            f"Error: {e}"
        )


def _fetch_sample() -> str:
    """Fetch the first 3 rows so the coder sees real values."""
    try:
        rows = requests.get(f"{API_BASE}/records", params={"limit": 3}, timeout=5).json()
        data = rows.get("data", [])
        return "\n".join(str(r) for r in data)
    except Exception:
        return "(sample unavailable)"


def _fetch_tags() -> str:
    """Fetch all tag names with counts so the coder knows exact casing."""
    try:
        tags = requests.get(f"{API_BASE}/tags", timeout=5).json().get("tags", [])
        return "\n".join(f"  - \"{t['tag_nm']}\" ({t['count']} rows)" for t in tags)
    except Exception:
        return "(tags unavailable)"


def create_coder_agent(model: GeminiModel) -> Agent:
    schema_str = _fetch_schema()
    sample_str = _fetch_sample()
    tags_str = _fetch_tags()

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
        system_prompt=(
            "You are an expert Python data analyst and engineer. "
            "You will receive one of two types of input:\n\n"
            "**Type 1 — New task (plain English)**\n"
            "Write self-contained, executable Python code that performs the analysis.\n\n"
            "**Type 2 — Retry (fix a failed execution)**\n"
            "Input format:\n"
            "  TASK: <original instruction>\n"
            "  ERROR: <traceback or error output from the previous execution>\n"
            "  PREVIOUS CODE: <the code that produced the error>\n"
            "Diagnose the error, fix it, and return the corrected code.\n\n"
            "## Data Source\n"
            f"Base URL: {API_BASE}\n\n"
            "Available endpoints:\n"
            "- GET /records       → all rows. Query params: tag_nm (partial), account_id (int), "
            "accountnumber (str), limit (int), offset (int). Response key: 'data'\n"
            "- GET /tags          → unique tag names + counts. Response key: 'tags'\n"
            "- GET /accounts      → unique customer accounts (deduplicated by accountnumber). "
            "Response keys: 'total_unique_accounts' (int), 'accounts' (list of {accountnumber})\n"
            "- GET /accounts/{account_id} → all tags for one account. Response key: 'records'\n\n"
            "## Live Schema (fetched from API)\n"
            f"{schema_str}\n\n"
            "## Sample rows (for reference)\n"
            f"{sample_str}\n\n"
            "## All tag names in the dataset (exact casing)\n"
            f"{tags_str}\n\n"
            "## Rules\n"
            "- ALWAYS fetch data from the API using `requests`. Never invent or hardcode data.\n"
            "- Use `pd.DataFrame(requests.get(f'{API_BASE}/records').json()['data'])` to load all rows.\n"
            "- When filtering by tag_nm, ALWAYS use case-insensitive comparison: "
            "`df[df['tag_nm'].str.lower() == target.lower()]`. Never use exact `==` on raw strings.\n"
            "- Use the exact tag names listed above when the user refers to a tag by name.\n"
            "- All imports must be included. Code must run as-is with no missing variables.\n"
            "- The FINAL output must be a single JSON object printed with:\n"
            "    import json; print(json.dumps(result))\n"
            "  where `result` is a dict with exactly two keys:\n"
            "    'summary': a one-line string describing what was computed\n"
            "    'data': the actual result (a number, list of dicts, or dict)\n"
            "  Example: print(json.dumps({'summary': 'Top 3 tags by count', "
            "'data': [{'tag_nm': 'Charged Off', 'count': 5}]}))\n"
            "- Do NOT print anything else — only the single JSON line.\n"
            "- Return ONLY raw Python code — no markdown fences, no prose."
        ),
    )
