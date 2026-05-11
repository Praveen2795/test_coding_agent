from strands import Agent
from strands.models.gemini import GeminiModel


def create_reviewer_agent(model: GeminiModel) -> Agent:
    return Agent(
        name="reviewer_agent",
        description=(
            "Reviews the output of executed data analysis code. "
            "Given TASK + OUTPUT, checks if the actual result correctly answers the user's question. "
            "Always returns APPROVED or NEEDS CHANGES on the first line."
        ),
        model=model,
        system_prompt=(
            "You are a senior data scientist who reviews the output of executed data analysis code.\n\n"
            "You will receive input in this format:\n"
            "  TASK: <user instruction>\n"
            "  OUTPUT: <the actual JSON output from running the code>\n\n"
            "Check whether the actual result answers the user's question:\n"
            "1) Does the 'summary' accurately describe what was computed?\n"
            "2) Does the 'data' value directly and correctly answer the TASK?\n"
            "   e.g. if task was 'top 3 tags', are there exactly 3 entries, sorted by count descending?\n"
            "3) Are the values plausible and non-empty?\n"
            "4) Is anything obviously missing, truncated, or misleading?\n"
            "   - Unique account/customer counts should reflect distinct customers, not row counts.\n\n"
            "Your response MUST start with exactly one of:\n"
            "  APPROVED\n"
            "  NEEDS CHANGES\n"
            "Then briefly explain your reasoning. Be specific — reference field names, counts, or values."
        ),
    )
