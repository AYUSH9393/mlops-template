import logging


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", '
        '"level": "%(levelname)s", '
        '"event": "%(message)s"}'
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
