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
        return False, f"No valid keys in data, expect of: ({LEGACY_KEYS})"

    if sum([len(data[key]) for key in LEGACY_KEYS]) == 0:
        return False, "all valid keys in data are empty"

    return True, ""


def check_message(data):
    if len(data) == 0:
        return False, "messages should not be empty"

    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        return False, f"messages should be {FMT_MESSAGES}, but got {data}"

    if data[0]["role"] not in ("system", "user"):
        return False, f"messages[0]['role'] should be 'system' or 'user', but got {data[0]['role']}"

    if data[-1]["role"] != "assistant":
        return False, f"messages[-1]['role'] should be 'assistant', but got {data[-1]['role']}"

    if data[-2]["role"] != "user":
        return False, "messages[-2]['role'] should be 'user'"

    if data[0]["role"] == "system" and len(data[1:]) % 2 != 0:
        return False, f"messages should be in pairs between user and assistant, but got {data}"

    if any([len(d["content"]) == 0 for d in data]):
        return False, f"messages should not be empty, but some of them are empty. see {data}"

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
