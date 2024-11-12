#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


from hakkero.dataset.strategy.errors import SegmentationError
from hakkero.dataset.strategy.errors import TokenizationError
from hakkero.dataset.strategy.segmentation import concat
from hakkero.dataset.strategy.segmentation import integrous
from hakkero.dataset.strategy.segmentation import naive
from hakkero.dataset.strategy.segmentation import unbiased
from hakkero.dataset.strategy.tokenization import chatml_message
from hakkero.dataset.strategy.tokenization import chatml_preference
from hakkero.dataset.strategy.tokenization import chatml_qwen2_vl_message
from hakkero.dataset.strategy.tokenization import huggingface_message
from hakkero.dataset.strategy.tokenization import huggingface_preference
from hakkero.dataset.strategy.tokenization import legacy

ST_INTEGROUS = "integrous"
ST_CONCAT = "concat"
ST_NAIVE = "naive"
ST_UNBIASED = "unbiased"

segment = {
    ST_INTEGROUS: integrous,
    ST_CONCAT: concat,
    ST_NAIVE: naive,
    ST_UNBIASED: unbiased,
}


ST_LEGACY = "legacy"
ST_HG = "hg"
ST_HG_PREFERENCE = "hg_preference"
ST_CHATML = "chatml"
ST_CHATML_PREFERENCE = "chatml_preference"
ST_CHATML_QWEN_VL = "chatml_qwen2_vl_message"

tokenize = {
    ST_LEGACY: legacy,
    ST_HG: huggingface_message,
    ST_HG_PREFERENCE: huggingface_preference,
    ST_CHATML: chatml_message,
    ST_CHATML_PREFERENCE: chatml_preference,
    ST_CHATML_QWEN_VL: chatml_qwen2_vl_message,
}


ST_SEGMENT = "st_segment"
ST_TOKENIZE = "st_tokenize"
default_strategy = {ST_SEGMENT: ST_NAIVE, ST_TOKENIZE: ST_LEGACY}
