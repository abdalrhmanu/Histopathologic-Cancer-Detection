try:
    # https://pypi.org/project/loguru/
    from loguru import logger
except ImportError:

    import logging

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("pipeline")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
