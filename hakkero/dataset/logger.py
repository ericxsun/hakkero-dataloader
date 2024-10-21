#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


import logging
import logging.handlers
import os

logger = logging.getLogger(__name__)


FMT_LOG = "[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d] %(message)s"
FMT_DATE = "%Y-%m-%d %H:%M:%S"


def _configure_logger(level="INFO", filename=None):
    formatter = logging.Formatter(fmt=FMT_LOG, datefmt=FMT_DATE)

    handler_console = logging.StreamHandler()
    handler_console.setLevel(level)
    handler_console.setFormatter(formatter)
    logger.addHandler(handler_console)

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        handler_file = logging.handlers.RotatingFileHandler(filename, maxBytes=5 * 1024 * 1024, backupCount=2)
        handler_file.setLevel(level)
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    logger.propagate = False


lvl = os.environ.get("HAKKERO_LOG_LEVEL", "INFO")
log = os.environ.get("HAKKERO_LOG_FILE", None)

_configure_logger(lvl, log)
