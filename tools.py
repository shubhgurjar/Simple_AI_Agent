from datetime import datetime
import json
import logging

import requests


class AgentTools:
    """Provide the functions and schemas exposed to the agent as tools."""

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "get_current_date_time",
                "description": (
                    "Returns the current date and time as a string in the "
                    "format 'YYYY-MM-DD HH:MM:SS'."
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
                "description": "Returns the current weather for a given city.",
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
    ]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.functions = {
            "get_current_date_time": self.get_current_date_time,
            "get_current_weather": self.get_current_weather,
            "get_user_location": self.get_user_location,
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
                self.logger.info(
                    f"Executing tool: name={tool_name} arguments={tool_args!r}"
                )
                tool_result = tool_function(**tool_args)
                tool_content = str(tool_result)
                self.logger.info(
                    f"Tool completed: name={tool_name} result={tool_content!r}"
                )
            except Exception:
                self.logger.exception(
                    f"Tool failed: name={tool_name} "
                    f"arguments={tool_call.function.arguments}"
                )
                raise

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
            city = response.json().get("city", "Unknown city")
            self.logger.info(f"User city retrieved: {city!r}")
            return city

        self.logger.warning(
            f"Location service returned status={response.status_code}"
        )
        return "Could not retrieve user location."
