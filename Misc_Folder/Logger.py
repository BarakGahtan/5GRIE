import sys
from os import environ

from loguru import logger

def setup_logger() -> None:
    """Set up stderr logging format.

    The logging format and colors can be overridden by setting up the
    environment variables such as ``LOGURU_FORMAT``.
    See `Loguru documentation`_ for details.

    .. _Loguru documentation: https://loguru.readthedocs.io/en/stable/api/logger.html#env
    """
    logger.remove()  # Remove the default setting

    # Set up the preferred logging colors and format unless overridden by its environment variable
    logger.level("INFO", color=environ.get("LOGURU_INFO_COLOR") or "<white>")
    logger.level("DEBUG", color=environ.get("LOGURU_DEBUG_COLOR") or "<d><white>")
    log_format = environ.get("LOGURU_FORMAT") or (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "<b><level>{level: <8}</level></b> "
        "| <level>{message}</level>"
    )
    logger.add(sys.stderr, format=log_format)

    # By default all the logging messages are disabled
    logger.enable("charger")