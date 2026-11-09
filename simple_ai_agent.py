import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from Utils import configure_logging
from tools import AgentTools

MAX_STEPS = 5
agent_tools = AgentTools()


class SimpleAIAgent:
    """Run a chat session with an OpenAI-compatible API and local tools."""

    def __init__(self, debug_enabled=False):
        # Load API settings from the project's .env file before creating the client.
        load_dotenv()
        self.debug_enabled = debug_enabled
        self.client = OpenAI(
            base_url=os.environ["BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
        self.model = os.environ["MODEL_NAME"]
        self.logger = logging.getLogger(__name__)
        self.messages = [
            {
                "role": "system",
                "content": "You are a professional agent. Always reply within 100 words.",
            }
        ]

    def run(self):
        """Read user turns and handle them until the user exits."""
        while True:
            prompt = "🧑 You: "
            if sys.stdout.isatty():
                prompt = f"\033[97m{prompt}\033[0m"
            user_input = input(prompt)
            if user_input.lower() in {"exit", "quit", "bye"}:
                break

            self.logger.info(f"🧑 User asked: {user_input}")
            self.messages.append({"role": "user", "content": user_input})
            final_message = self._get_agent_response()
            if self.debug_enabled:
                self.logger.info(f"🤖 Agent: {final_message.content}")
            else:
                print(f"🤖 Agent: {final_message.content}")

    def _get_agent_response(self):
        """Run model and tool-call turns until the model returns a final reply."""
        for _ in range(MAX_STEPS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=agent_tools.schemas,
            )
            choice = response.choices[0]
            message = choice.message
            self.logger.info(
                f"🤖 LLM response received: "
                f"finish_reason={choice.finish_reason} content={message.content!r}"
            )

            # Keep the agent message so tool calls remain in the conversation.
            self.messages.append(message)
            if not message.tool_calls:
                return message

            for tool_call in message.tool_calls:
                self.messages.append(agent_tools.execute_tool_call(tool_call))

        raise RuntimeError(
            f"Agent exceeded the maximum of {MAX_STEPS} model/tool turns."
        )


def main():
    load_dotenv()
    debug_enabled = configure_logging()
    SimpleAIAgent(debug_enabled=debug_enabled).run()


if __name__ == "__main__":
    main()
