#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

KEEP_THINK_ALL = "all"
KEEP_THINK_LAST = "last"
KEEP_THINK_NO = "no"
keep_think_strategy = [KEEP_THINK_NO, KEEP_THINK_ALL, KEEP_THINK_LAST]


KEEP_THINK_REMAP = {
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}": "{% if (not keep_think is defined or not keep_think) and '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}",
    "{%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}": "{%- set content = message.content %}{% if (not keep_think is defined or not keep_think) and '</think>' in content %}{% set content = message.content.split('</think>')[-1].lstrip('\\n') %}{% endif %}",
}


def change_chat_template_with_keep_think_remap(chat_template, keep_think):
    new_chat_template = chat_template

    if keep_think in {KEEP_THINK_ALL, KEEP_THINK_LAST}:
        for key, value in KEEP_THINK_REMAP.items():
            new_chat_template = new_chat_template.replace(key, value)
    else:
        keep_think = KEEP_THINK_NO

    return new_chat_template, keep_think
