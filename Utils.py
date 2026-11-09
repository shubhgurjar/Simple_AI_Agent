import logging
import os
import sys


class GrayIndentedFormatter(logging.Formatter):
    """Style detailed logs gray and conversation turns white in a terminal."""

    def format(self, record):
        message = super().format(record)
        if message.startswith(("🧑 User asked:", "🤖 Agent:")):
            if sys.stderr.isatty():
                return f"\033[97m{message}\033[0m"
            return message

        if sys.stderr.isatty():
            return f"\033[90m    {message}\033[0m"
        return f"    {message}"


def configure_logging():
    """Enable INFO logs only when the DEBUG environment variable is true."""
    debug_enabled = os.getenv("DEBUG", "false").strip().lower() == "true"
    log_level = logging.INFO if debug_enabled else logging.CRITICAL + 1
    logging.basicConfig(level=log_level, format="%(message)s")
    logging.getLogger().setLevel(log_level)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(GrayIndentedFormatter("%(message)s"))
    return debug_enabled
