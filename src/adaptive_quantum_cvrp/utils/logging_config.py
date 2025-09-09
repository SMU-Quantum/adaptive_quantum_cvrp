# src/adaptive_quantum_cvrp/utils/logging_config.py

"""
provides a single function to set up a professional logging system. It will log messages to both the console and a file, with clear,
 timestamped formatting.
"""

import logging
import sys
from pathlib import Path

def setup_logging(log_dir: Path, log_level: str = "INFO") -> None:
    """
    Configures the root logger for the application.

    This setup directs log messages to both a file and the console.

    Args:
        log_dir: The directory where the log file will be saved.
        log_level: The minimum logging level to capture (e.g., "INFO", "DEBUG").
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiment.log"

    # Define the format for log messages
    log_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] - %(message)s"
    )
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level.upper())
    
    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a handler to write logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Create a handler to stream logs to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)


    logging.getLogger("stevedore").setLevel(logging.WARNING)
    logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
    logging.info(f"Logging configured. Log file at: {log_file}")