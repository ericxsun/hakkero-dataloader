#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import glob
import hashlib
import json
import os.path
import shutil
from pathlib import Path

import numpy as np


def object_to_dict(obj):
    return dict((key, value) for key, value in obj.__dict__.items() if not callable(value) and not key.startswith("_"))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, tuple)):
            return list(obj)

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return object_to_dict(obj)


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as fin:
        return json.load(fin)


def save_json(data, filename):
    with open(filename, "w") as fd:
        json.dump(data, fd, ensure_ascii=False, indent=2)


def load_jsonl(filename):
    paths = filename
    if isinstance(paths, str):
        paths = [paths]

    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend([os.path.join(dp, f) for dp, _, fn in os.walk(os.path.expanduser(path)) for f in fn])
        else:
            files.extend([path])

    data = []
    for filename in files:
        with open(filename, "r", encoding="utf-8") as fin:
            data.extend([json.loads(line) for line in fin])

    return data


def save_jsonl(data, filename):
    with open(filename, "w") as fd:
        for line in data:
            fd.write(json.dumps(line, ensure_ascii=False) + "\n")


def special_copy(src, dst, exts=None):
    if src is None or dst is None or not os.path.exists(src):
        return

    if exts is None:
        exts = []

    filenames = []
    for ext in exts:
        filenames.extend(glob.glob(os.path.join(src, f"*{ext}")))

    for filename in filenames:
        dst_file = Path(dst) / Path(filename).name
        try:
            shutil.copy(filename, dst_file)
        except shutil.SameFileError:
            pass


def get_md5(text):
    if isinstance(text, str):
        text = text.encode()
    elif isinstance(text, (dict, list, tuple)):
        text = json.dumps(text, ensure_ascii=False).encode()

    return hashlib.md5(text).hexdigest()
