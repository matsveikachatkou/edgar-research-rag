"""
agents/agent.py — Base class for all agents.
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class Agent:
    """Base class providing logging and shared config for all agents."""

    # ANSI color codes for terminal output
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"

    name: str = "Agent"
    color: str = RESET

    def __init__(self):
        self.logger = logging.getLogger(self.name)

    def log(self, message: str) -> None:
        """Log a colored message to stdout."""
        self.logger.info(self.color + message + self.RESET)