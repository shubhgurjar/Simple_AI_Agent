# Simple AI Agent

A command-line AI agent powered by an OpenAI-compatible chat completions API. It keeps conversation history during the session and supports tools for the current date and time, weather by city, and IP-based city lookup.

## Requirements

- Python 3
- An API endpoint and model that support chat completions and function tools
- Internet access for the weather tool

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, activate it with:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. Install the dependencies from the requirements file:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Create a `.env` file in the project directory with your API settings:

   ```dotenv
   BASE_URL=https://your-api-endpoint/v1
   API_KEY=your-api-key
   MODEL_NAME=your-model-name
   DEBUG=false
   ```

   Use the base URL and model name required by your provider. Keep `.env` private and do not commit real API credentials.

## Run

From the project directory, run:

```bash
python simple_ai_agent.py
```

Enter a prompt at `You:`. Type `exit`, `quit`, or `bye` to close the agent.

## Available tools

- `get_current_date_time` returns the local date and time in `YYYY-MM-DD HH:MM:SS` format.
- `get_current_weather` looks up a city using wttr.in and returns a short weather description.
- `get_user_location` looks up the city associated with the current public IP using ipinfo.io.

Tool definitions and implementations are in [tools.py](tools.py). The agent's chat and tool-dispatch loop is in [simple_ai_agent.py](simple_ai_agent.py).

## Logging

Set `DEBUG=true` in `.env` (or in the shell environment) to show INFO logs, including LLM and tool details. Set it to `false` or leave it unset to hide all logs. The agent's final answer is still printed when logging is disabled. Logs may contain prompts, tool arguments, or tool output; avoid entering sensitive information.

## Tests

Run the test suite with Python's built-in unittest runner:

```bash
python -m unittest -v
```

The tests mock external HTTP calls so they can run without contacting the weather or location services.
