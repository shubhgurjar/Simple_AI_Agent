import logging
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from Utils import GrayIndentedFormatter, configure_logging
from simple_ai_agent import MAX_STEPS, SimpleAIAgent
from tools import AgentTools


def make_tool_call(name, arguments="{}", call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_completion(message, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        self.tools = AgentTools()

    def test_get_current_date_time_returns_formatted_time(self):
        with patch("tools.datetime") as mocked_datetime:
            mocked_datetime.now.return_value.strftime.return_value = (
                "2026-01-02 03:04:05"
            )
            result = self.tools.get_current_date_time()

        self.assertEqual(result, "2026-01-02 03:04:05")

    @patch("tools.requests.get")
    def test_get_current_weather_returns_response_text(self, mock_get):
        mock_get.return_value = SimpleNamespace(
            status_code=200,
            text="London: Cloudy",
        )

        result = self.tools.get_current_weather("London")

        self.assertEqual(result, "London: Cloudy")
        mock_get.assert_called_once_with("https://wttr.in/London?format=3")

    @patch("tools.requests.get")
    def test_get_current_weather_reports_http_failure(self, mock_get):
        mock_get.return_value = SimpleNamespace(status_code=503, text="")

        result = self.tools.get_current_weather("London")

        self.assertEqual(
            result,
            "Could not retrieve weather information for London.",
        )

    @patch("tools.requests.get", side_effect=requests.RequestException("offline"))
    def test_get_current_weather_reraises_request_error(self, _mock_get):
        with self.assertRaises(requests.RequestException):
            self.tools.get_current_weather("London")

    @patch("tools.requests.get")
    def test_get_user_location_returns_city(self, mock_get):
        mock_get.return_value = SimpleNamespace(
            status_code=200,
            json=lambda: {"city": "Toronto"},
        )

        self.assertEqual(self.tools.get_user_location(), "Toronto")
        mock_get.assert_called_once_with("https://ipinfo.io/json")

    @patch("tools.requests.get")
    def test_get_user_location_defaults_when_city_is_missing(self, mock_get):
        mock_get.return_value = SimpleNamespace(status_code=200, json=dict)

        self.assertEqual(self.tools.get_user_location(), "Unknown city")

    @patch("tools.requests.get")
    def test_get_user_location_reports_http_failure(self, mock_get):
        mock_get.return_value = SimpleNamespace(status_code=503)

        self.assertEqual(
            self.tools.get_user_location(),
            "Could not retrieve user location.",
        )

    @patch("tools.requests.get", side_effect=requests.RequestException("offline"))
    def test_get_user_location_reraises_request_error(self, _mock_get):
        with self.assertRaises(requests.RequestException):
            self.tools.get_user_location()

    def test_execute_tool_call_invokes_registered_tool(self):
        self.tools.functions["get_current_weather"] = Mock(return_value="Paris: Clear")
        result = self.tools.execute_tool_call(
            make_tool_call("get_current_weather", '{"city": "Paris"}')
        )

        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "call-1")
        self.assertEqual(result["content"], "Paris: Clear")

    def test_execute_tool_call_reports_unregistered_tool(self):
        result = self.tools.execute_tool_call(make_tool_call("missing_tool"))

        self.assertEqual(result["role"], "tool")
        self.assertIn("not registered", result["content"])

    def test_execute_tool_call_reraises_invalid_json(self):
        with self.assertRaises(ValueError):
            self.tools.execute_tool_call(
                make_tool_call("get_current_date_time", "not-json")
            )


class AgentTests(unittest.TestCase):
    def make_agent(self, client):
        with patch.dict(
            "os.environ",
            {
                "BASE_URL": "https://example.test/v1",
                "API_KEY": "test-key",
                "MODEL_NAME": "test-model",
            },
        ), patch("simple_ai_agent.load_dotenv"), patch(
            "simple_ai_agent.OpenAI", return_value=client
        ):
            return SimpleAIAgent()

    def test_response_returns_message_when_no_tool_is_requested(self):
        final_message = SimpleNamespace(content="Hello", tool_calls=None)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(return_value=make_completion(final_message))
                )
            )
        )
        agent = self.make_agent(client)

        result = agent._get_agent_response()

        self.assertIs(result, final_message)
        self.assertEqual(agent.messages[-1], final_message)

    def test_response_executes_tool_then_requests_final_message(self):
        tool_call = make_tool_call("get_current_date_time")
        tool_message = SimpleNamespace(content=None, tool_calls=[tool_call])
        final_message = SimpleNamespace(content="It is now.", tool_calls=None)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        side_effect=[
                            make_completion(tool_message, "tool_calls"),
                            make_completion(final_message),
                        ]
                    )
                )
            )
        )
        agent = self.make_agent(client)

        result = agent._get_agent_response()

        self.assertIs(result, final_message)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(agent.messages[-2]["role"], "tool")
        self.assertEqual(agent.messages[-2]["tool_call_id"], "call-1")

    def test_response_raises_after_maximum_tool_turns(self):
        tool_message = SimpleNamespace(
            content=None,
            tool_calls=[make_tool_call("get_current_date_time")],
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        return_value=make_completion(tool_message, "tool_calls")
                    )
                )
            )
        )
        agent = self.make_agent(client)

        with self.assertRaisesRegex(RuntimeError, str(MAX_STEPS)):
            agent._get_agent_response()

    def test_run_stops_on_exit_command(self):
        agent = self.make_agent(Mock())
        with patch("builtins.input", side_effect=["exit"]):
            agent.run()

        self.assertEqual(len(agent.messages), 1)


class LoggingTests(unittest.TestCase):
    def test_conversation_messages_are_not_indented_outside_terminal(self):
        formatter = GrayIndentedFormatter("%(message)s")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "🤖 Agent: hello", (), None
        )

        with patch("Utils.sys.stderr.isatty", return_value=False):
            result = formatter.format(record)

        self.assertEqual(result, "🤖 Agent: hello")

    def test_detailed_logs_are_indented_outside_terminal(self):
        formatter = GrayIndentedFormatter("%(message)s")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "Tool completed", (), None
        )

        with patch("Utils.sys.stderr.isatty", return_value=False):
            result = formatter.format(record)

        self.assertEqual(result, "    Tool completed")

    def test_debug_true_enables_info_logging(self):
        root_logger = logging.getLogger()
        original_level = root_logger.level
        try:
            with patch.dict("os.environ", {"DEBUG": "true"}), patch(
                "Utils.logging.basicConfig"
            ):
                enabled = configure_logging()
            self.assertTrue(enabled)
            self.assertEqual(root_logger.level, logging.INFO)
        finally:
            root_logger.setLevel(original_level)

    def test_debug_false_hides_all_logs(self):
        root_logger = logging.getLogger()
        original_level = root_logger.level
        try:
            with patch.dict("os.environ", {"DEBUG": "false"}), patch(
                "Utils.logging.basicConfig"
            ):
                enabled = configure_logging()
            self.assertFalse(enabled)
            self.assertEqual(root_logger.level, logging.CRITICAL + 1)
        finally:
            root_logger.setLevel(original_level)


if __name__ == "__main__":
    unittest.main()
