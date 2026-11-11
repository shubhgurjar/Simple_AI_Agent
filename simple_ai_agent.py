import logging
import os
import sys
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from utils import configure_logging
from tools import AgentTools

MAX_STEPS = 5
agent_tools = AgentTools()
# SYSTEM_PROMPT = """You are a helpful, professional AI Agent. Be clear, respectful,
# accurate, and practical. Keep each response to 200 words or fewer. Answer directly,
# using concise bullets or steps when useful. Ask a brief clarifying question when
# needed. Never invent facts or tool results; state uncertainty and limitations plainly.

# Use tools when the request needs live information or local knowledge:
# - `get_current_date_time`: provides current local date and time.
# - `get_current_weather`: provides current weather for a specified city.
# - `get_user_location`: provides user's city based on their IP address.
# - `search_knowledge_base`: always search first for company-specific inquiries, such as
#   questions about Zorvex Company Private Limited, hiring, HR, policies (including medical
#   policies), RC documents, technical documents, or travel expenses. Treat related
#   wording as in-scope even when the company name is omitted. For every new company
#   query or follow-up, including changed details or a different employee group, run a
#   fresh search before answering; never reuse earlier search results. Make short
#   follow-ups self-contained using relevant conversation context (for example, search
#   for maternity leave for mothers when the user asks a contextual follow-up). Review the
#   three returned sections and use only the ones that actually answer the question.
#   Treat them as internal reference material: synthesize a concise answer and do not
#   reproduce or list the chunks. If the search fails or no relevant information is
#   found, say so and do not infer company policies from general knowledge.

# For questions outside those company-specific topics, answer from your general training
# knowledge. Treat successful live-tool results as authoritative for current date, time,
# weather, and location. Report only relevant details returned by the tool. Never call a
# tool result simulated or outdated, contradict it with training-data assumptions, add
# unrelated date caveats to a weather answer, or speculate beyond the result. If a tool
# reports an error or uncertainty, state that plainly. When freshness matters and no live
# tool verifies the answer, acknowledge uncertainty. Avoid generic closing filler."""

SYSTEM_PROMPT = """You are a helpful, professional AI Agent. Reply within 200 words. Few similary queries might need same tool calling again and again so do not hesitate to call tools."""

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
                "content": SYSTEM_PROMPT,
            }
        ]
    

    def run(self):
        """Read user turns and handle them until the user exits."""
        try:
            while True:
                prompt = "🧑 You: "
                if sys.stdout.isatty():
                    prompt = f"\033[97m{prompt}\033[0m"
                user_input = input(prompt)
                if user_input.lower() in {"exit", "quit", "bye"}:
                    break

                self.logger.info(f"🧑 User asked: {user_input}")
                self.messages.append({"role": "user", "content": user_input})
                try:
                    self._get_agent_response()
                except OpenAIError as exc:
                    self.logger.exception("Model request failed")
                    print(f"\n🤖 Agent: Model request failed: {exc}")
                except RuntimeError as exc:
                    self.logger.error("Agent stopped: %s", exc)
                    print(f"\n🤖 Agent: {exc}")
        except KeyboardInterrupt:
            self.logger.info("Agent session interrupted by user.")
            print("\n🤖 Agent: Interrupted. Goodbye.")

    def _process_stream(self, history_messages):
        """Stream the model's response and tool calls."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=history_messages,
            tools=agent_tools.schemas,
            stream=True
        )

        full_response = ""
        tool_calls = {}
        label_printed = False
        for event in stream:

            if not event.choices:
                continue

            delta = event.choices[0].delta

            # Flush each content chunk immediately so the response appears live.
            if delta.content:
                if not label_printed:
                    print("🤖 Agent: ", end="", flush=True)
                    label_printed = True
                print(delta.content, end="", flush=True)
                full_response += delta.content

            # Tool calls arrive in fragments; the index groups each call's pieces.
            if delta.tool_calls:
                for tc in (delta.tool_calls or []):
                    tool_calls[tc.index] = tool_calls.setdefault(
                        tc.index,
                        SimpleNamespace(
                            id=None,
                            function=SimpleNamespace(name="", arguments=""),
                        ),
                    )
                    obj = tool_calls[tc.index]

                    # The ID usually arrives once; name and arguments may span chunks.
                    if tc.id:
                        obj.id = tc.id

                    if tc.function and tc.function.name:
                        obj.function.name  += tc.function.name

                    if tc.function and tc.function.arguments:
                        obj.function.arguments += tc.function.arguments

        if full_response:
            print()

        # Return calls in their original order, not the order their chunks arrived.
        calls_list = []
        for index in sorted(tool_calls):
            calls_list.append(tool_calls[index])

        return full_response, calls_list

    def _build_tool_call_payloads(self, tool_calls):
        """Convert tool call deltas into a list of payloads for the model."""
        payloads = []
        for tc in tool_calls:
            tool_dict = {}
            tool_dict["id"] = tc.id
            # Chat Completions expects function calls in this nested message shape.
            tool_dict["type"] = "function"
            tool_dict["function"] = {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            payloads.append(tool_dict)
        return payloads


    def _get_agent_response(self):
        """Run model and tool-call turns until the model returns a final reply."""
        # After each tool result, request another model turn to finish the reply.
        for _ in range(MAX_STEPS):
            full_response, tool_calls = self._process_stream(self.messages)

            message = {
                "role": "assistant",
                "content": full_response,
            }
            if tool_calls:
                message["tool_calls"] = self._build_tool_call_payloads(tool_calls)
            # Keep the assistant's tool-call message before its matching tool results.
            self.messages.append(message)
            if not tool_calls:
                return message

            for tool_call in tool_calls:
                result = agent_tools.execute_tool_call(tool_call)
                self.messages.append(result)
        else:
            raise RuntimeError(
                f"Agent exceeded the maximum of {MAX_STEPS} model/tool turns."
            )


def main():
    load_dotenv()
    debug_enabled = configure_logging()
    try:
        SimpleAIAgent(debug_enabled=debug_enabled).run()
    except KeyError as exc:
        logging.error("Missing required environment variable: %s", exc.args[0])
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
