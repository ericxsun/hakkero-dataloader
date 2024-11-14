#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


LEGACY_KEYS = ["title", "summary", "abstract", "text", "question", "answer", "code"]
FMT_MESSAGES = "[{'role': 'x', 'content': 'x'}, ...]"
FMT_PREFERENCES = "{'context': [{'role': 'x', 'content': 'x'}, ...], 'chosen': 'x', 'rejected': 'x'}"


def check_legacy(data):
    if not isinstance(data, dict):
        return False, "wrong data format, expect {key: value}, " + f"but got {data}"

    if all(s not in data for s in LEGACY_KEYS):
        return False, f"No valid keys in input, expect of: ({LEGACY_KEYS})"

    return True, ""


def check_message(data):
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        return False, f"messages should be {FMT_MESSAGES}, but got {data}"

    if data[-1]["role"] != "assistant":
        return False, "messages[-1]['role'] should be 'assistant'"

    if data[-2]["role"] != "user":
        return False, "messages[-2]['role'] should be 'user'"

    return True, ""


def check_preference(data):
    if not isinstance(data, dict):
        return False, f"messages should be {FMT_PREFERENCES}, but got {data}"

    if "context" not in data or "chosen" not in data or "rejected" not in data:
        return False, f"messages should be {FMT_PREFERENCES}, but got {data}"

    if not isinstance(data["chosen"], str) or not isinstance(data["rejected"], str):
        return False, "chosen or rejected should be string"

    return check_message(data["context"])


check_func = {
    "legacy": check_legacy,
    "message": check_message,
    "preference": check_preference,
}
