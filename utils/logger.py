#!/usr/bin/env python3
"""
Logging configuration for the pipeline
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console

console = Console()


def setup_logger(
    name: str = "LoS-Audit",
    log_dir: str = "logs",
    level: str = "INFO",
    save_to_file: bool = True
) -> logging.Logger:
    """
    Setup logger with rich formatting and file output
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        save_to_file: Whether to save logs to file
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)
    
    # File handler
    if save_to_file:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def log_section(logger: logging.Logger, title: str):
    """Log a section header"""
    separator = "=" * 80
    logger.info(f"\n{separator}")
    logger.info(f"  {title}")
    logger.info(f"{separator}\n")


def log_config(logger: logging.Logger, config: dict):
    """Log configuration in a readable format"""
    logger.info("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")
