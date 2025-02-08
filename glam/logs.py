import logging

logger_name = "GLAM"


def setup_logger(log_level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(logger_name)
