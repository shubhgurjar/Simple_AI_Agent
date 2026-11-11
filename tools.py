from datetime import datetime
import json
import logging

import requests

from rag import SimpleRAGPipeline


class AgentTools:
    """Provide the functions and schemas exposed to the agent as tools."""

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "get_current_date_time",
                "description": (
                    "Returns the actual local date and time at the time of the "
                    "request in 'YYYY-MM-DD HH:MM:SS' format. Treat this live "
                    "result as authoritative."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": (
                    "Returns current weather for a given city. Report the result "
                    "as returned; do not add unsupported conditions or unrelated "
                    "date caveats."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city to get the weather for.",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_location",
                "description": "Returns the user's city based on their IP address.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": (
                    "Search the company-only knowledge base for Zorvex Company "
                    "Private Limited information. It returns the three most "
                    "relevant candidate sections. Call it again for every new "
                    "company-related question or follow-up; do not rely on earlier "
                    "search results. Compare the candidates and use only sections "
                    "that answer the question; treat them as internal reference "
                    "material, not text to repeat in the final answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": (
                                "A self-contained search query using the user's "
                                "current question and relevant conversation context. "
                                "The company name does not need to be included."
                            ),
                        }
                    },
                    "required": ["user_query"],
                },
            },
        },
    ]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._rag_pipeline = None
        self.functions = {
            "get_current_date_time": self.get_current_date_time,
            "get_current_weather": self.get_current_weather,
            "get_user_location": self.get_user_location,
            "search_knowledge_base": self.search_knowledge_base,
        }

    def execute_tool_call(self, tool_call):
        """Run a model-requested tool and return its chat-completions result."""
        tool_name = tool_call.function.name
        tool_function = self.functions.get(tool_name)
        if not tool_function:
            self.logger.warning(
                f"LLM requested an unregistered tool: "
                f"id={tool_call.id} name={tool_name}"
            )
            tool_content = f"Tool {tool_name!r} is not registered."
        else:
            try:
                # Decode the model arguments and invoke the matching tool method.
                tool_args = json.loads(tool_call.function.arguments)
                if not isinstance(tool_args, dict):
                    raise TypeError("Tool arguments must be a JSON object.")
            except (json.JSONDecodeError, TypeError) as exc:
                self.logger.warning(
                    f"Invalid arguments for tool: name={tool_name} error={exc}"
                )
                tool_content = f"Tool {tool_name!r} received invalid arguments."
            else:
                self.logger.info(
                    f"Executing tool: name={tool_name} arguments={tool_args!r}"
                )
                try:
                    tool_result = tool_function(**tool_args)
                except requests.RequestException as exc:
                    self.logger.warning(
                        f"Tool request failed: name={tool_name} error={exc}"
                    )
                    tool_content = (
                        f"Tool {tool_name!r} could not reach its external service."
                    )
                except TypeError as exc:
                    self.logger.warning(
                        f"Tool invocation failed: name={tool_name} error={exc}"
                    )
                    tool_content = f"Tool {tool_name!r} received invalid arguments."
                else:
                    tool_content = str(tool_result)
                    self.logger.info(
                        f"Tool completed: name={tool_name} result={tool_content!r}"
                    )

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_content,
        }

    def get_current_date_time(self):
        """Return the local date and time as YYYY-MM-DD HH:MM:SS."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"Current date and time retrieved: {current_time}")
        return current_time

    def get_current_weather(self, city):
        """Return the current weather description for a city."""
        url = f"https://wttr.in/{city}?format=3"
        self.logger.info(f"Requesting current weather for city={city!r}")

        try:
            response = requests.get(url)
        except requests.RequestException:
            self.logger.exception(f"Weather request failed for city={city!r}")
            raise

        if response.status_code == 200:
            weather = response.text.strip()
            self.logger.info(f"Weather retrieved for city={city!r}: {weather!r}")
            return weather

        self.logger.warning(
            f"Weather service returned status={response.status_code} "
            f"for city={city!r}"
        )
        return f"Could not retrieve weather information for {city}."

    def get_user_location(self):
        """Return the user's city based on their public IP address."""
        url = "https://ipinfo.io/json"
        self.logger.info("Requesting user location based on IP address")

        try:
            response = requests.get(url)
        except requests.RequestException:
            self.logger.exception("Location request failed")
            raise

        if response.status_code == 200:
            city = response.json().get("city", "")
            self.logger.info(f"User city retrieved: {city!r}")
            return city

        self.logger.warning(
            f"Location service returned status={response.status_code}"
        )
        return "Could not retrieve user location."

    def search_knowledge_base(self, user_query):
        """Return the three most relevant knowledge-base chunks for a query."""
        try:
            if self._rag_pipeline is None:
                self._rag_pipeline = SimpleRAGPipeline()
            results = self._rag_pipeline.search_knowledge_base(user_query, k=3)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            self.logger.exception("Knowledge-base search failed")
            return f"Knowledge-base search failed: {exc}"

        if not results:
            return "No knowledge-base sections were found."

        return "\n\n".join(
            (
                f"Source: {chunk['filename']} #{chunk['chunk_id']} "
                f"{chunk['chunk_text']}"
            )
            for chunk in results
        )
