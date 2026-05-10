from strands import Agent
from strands.models.gemini import GeminiModel
from strands_tools import python_repl

from agents.coder import create_coder_agent
from agents.reviewer import create_reviewer_agent


def create_supervisor_agent(model: GeminiModel) -> Agent:
    coder_agent = create_coder_agent(model)
    reviewer_agent = create_reviewer_agent(model)

    return Agent(
        name="supervisor_agent",
        description=(
            "Receives a plain-English data analysis instruction, orchestrates code generation, "
            "review, and execution, then returns the analysis result to the user."
        ),
        model=model,
        system_prompt=(
            "You are a data analysis supervisor. Users give you plain-English instructions "
            "like 'find the top 5 products by revenue' or 'show me the monthly sales trend'. "
            "You manage a pipeline with retry logic. Follow these steps exactly.\n\n"
            "**Step 1 — Generate code**\n"
            "Call `coder_agent` with the user's exact instruction. "
            "It will return raw Python code. Store that code exactly as returned.\n\n"
            "**Step 2 — Review**\n"
            "Call `reviewer_agent` with a message in this exact format:\n"
            "  TASK: <the original user instruction verbatim>\n"
            "  CODE: <the exact code returned by coder_agent>\n"
            "Check whether the response starts with 'APPROVED' or 'NEEDS CHANGES':\n"
            "  - APPROVED → go to Step 3.\n"
            "  - NEEDS CHANGES → stop. Tell the user the review failed and summarise the issues. "
            "Do NOT execute.\n\n"
            "**Step 3 — Execute**\n"
            "Call `python_repl` with the exact code from Step 1.\n"
            "  - SUCCESS (output contains valid JSON with 'summary' and 'data' keys) → go to Step 3b.\n"
            "  - ERROR (traceback, exception, or no valid JSON) → go to Step 3a.\n\n"
            "**Step 3a — Retry on execution error (max 2 retries total)**\n"
            "Call `coder_agent` again with this exact format:\n"
            "  TASK: <the original user instruction verbatim>\n"
            "  ERROR: <the full error output from python_repl>\n"
            "  PREVIOUS CODE: <the code that failed>\n"
            "Then repeat Step 2 (review) and Step 3 (execute) with the new code.\n"
            "If still failing after 2 retries, tell the user: "
            "'I was unable to generate working code for this task after 2 attempts.' "
            "Include a brief summary of the last error. Stop.\n\n"
            "**Step 3b — Post-execution output review**\n"
            "Call `reviewer_agent` with this exact format:\n"
            "  TASK: <the original user instruction verbatim>\n"
            "  OUTPUT: <the full JSON output from python_repl>\n"
            "Check whether the response starts with 'APPROVED' or 'NEEDS CHANGES':\n"
            "  - APPROVED → go to Step 4.\n"
            "  - NEEDS CHANGES → the output does not correctly answer the task. "
            "Call `coder_agent` with the retry format (TASK + ERROR=reviewer feedback + PREVIOUS CODE), "
            "then repeat Steps 2, 3, 3b. Count this as one retry (max 2 retries total across 3a and 3b).\n\n"
            "**Step 4 — Interpret and respond**\n"
            "Parse the JSON output (keys: 'summary', 'data') and present the result to the user "
            "in clear, friendly plain English:\n"
            "  - Start with a direct answer to the user's question.\n"
            "  - Present the data in a readable way (sentence, bullet list, or table as appropriate).\n"
            "  - Do NOT show raw JSON, code, or technical details to the user.\n\n"
            "You do not write, modify, or review code yourself. Always follow the pipeline."
        ),
        tools=[coder_agent, reviewer_agent, python_repl],
    )
