from strands import Agent
from strands.models.gemini import GeminiModel

from agents.coder import create_coder_agent


def create_supervisor_agent(model: GeminiModel) -> Agent:
    coder_agent = create_coder_agent(model)

    return Agent(
        name="supervisor_agent",
        description=(
            "Receives a plain-English data analysis instruction, delegates to the coder agent "
            "which writes and executes the code, then presents the result to the user."
        ),
        model=model,
        system_prompt=(
            "You are a data analysis supervisor. Users give you plain-English instructions "
            "like 'find the top 5 products by revenue' or 'show me the monthly sales trend'.\n\n"
            "**Step 1 — Generate and execute**\n"
            "Call `coder_agent` with the user's exact instruction. "
            "It will write, run, and fix code internally, then return the raw JSON result "
            "in the format: {\"summary\": \"...\", \"data\": ...}.\n\n"
            "**Step 2 — Interpret and respond**\n"
            "Parse the JSON output (keys: 'summary', 'data') and present the result to the user "
            "in clear, friendly plain English:\n"
            "  - Start with a direct answer to the user's question.\n"
            "  - Present the data in a readable way (sentence, bullet list, or table as appropriate).\n"
            "  - Do NOT show raw JSON, code, or technical details to the user.\n\n"
            "You do not write or modify code yourself. Always follow the pipeline."
        ),
        tools=[coder_agent],
    )
