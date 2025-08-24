"""Logging configuration and utilities for EpixVanity."""

import logging
import sys
from typing import Optional
from pathlib import Path

import colorama
from rich.logging import RichHandler
from rich.console import Console


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
    use_colors: bool = True
) -> logging.Logger:
    """Set up logging configuration for EpixVanity."""
    
    # Initialize colorama for cross-platform color support
    colorama.init()
    
    # Create logger
    logger = logging.getLogger("epix_vanity")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if use_rich:
        # Use Rich handler for beautiful console output
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True
        )
        formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]"
        )
    else:
        # Use standard console handler
        handler = logging.StreamHandler(sys.stderr)
        if use_colors:
            formatter = ColoredFormatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get the original formatted message
        message = super().format(record)
        
        # Add color based on log level
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        return f"{level_color}{message}{reset_color}"


def get_logger(name: str = "epix_vanity") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class ProgressLogger:
    """Logger for progress updates during vanity generation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize progress logger."""
        self.logger = logger or get_logger()
        self.last_log_time = 0
        self.log_interval = 10  # Log every 10 seconds
    
    def log_progress(
        self,
        attempts: int,
        rate: float,
        elapsed_time: float,
        pattern: str,
        force: bool = False
    ) -> None:
        """Log progress information."""
        import time
        
        current_time = time.time()
        if not force and current_time - self.last_log_time < self.log_interval:
            return
        
        self.logger.info(
            f"Progress: {attempts:,} attempts in {elapsed_time:.1f}s "
            f"({rate:.0f} attempts/s) - searching for '{pattern}'"
        )
        self.last_log_time = current_time
    
    def log_success(
        self,
        address: str,
        private_key: str,
        attempts: int,
        elapsed_time: float,
        pattern: str
    ) -> None:
        """Log successful vanity address generation."""
        self.logger.info(f"🎉 SUCCESS! Found vanity address matching '{pattern}'")
        self.logger.info(f"Address: {address}")
        self.logger.info(f"Private Key: {private_key}")
        self.logger.info(f"Found after {attempts:,} attempts in {elapsed_time:.1f}s")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log error information."""
        if context:
            self.logger.error(f"Error in {context}: {error}")
        else:
            self.logger.error(f"Error: {error}")
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
