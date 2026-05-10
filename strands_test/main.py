import os
from dotenv import load_dotenv
from strands.models.gemini import GeminiModel

from agents import create_supervisor_agent

load_dotenv()

# Show rich UI for tool calls in the terminal
os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"


def main():
    model = GeminiModel(
        model_id=os.environ["MODEL_NAME"],
        client_args={"api_key": os.environ["GOOGLE_API_KEY"]},
    )

    supervisor = create_supervisor_agent(model)

    print("=== Code Pipeline: Coder → Reviewer → Execute ===")
    print("Describe a coding task. The agent will write, review, and run it.")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            task = input("Task: ").strip()
            if not task:
                continue
            if task.lower() in ("exit", "quit"):
                break
            print()
            supervisor(task)
            print()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
