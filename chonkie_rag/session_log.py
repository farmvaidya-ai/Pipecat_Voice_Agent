from loguru import logger

SESSION_LOG_PATH = "session_log.log"

logger.add(
    SESSION_LOG_PATH,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} - {message}",
    level="DEBUG",
    encoding="utf-8",
    # No cap before this grew to 300+ MB unrotated. Rotate at 20 MB, keep the
    # last 5 rotated files so there's still real history to look back at.
    rotation="20 MB",
    retention=5,
)
