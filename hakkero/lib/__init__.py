#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import argparse
import os
import sys
from subprocess import check_output

ROOT_PATH = os.path.dirname(__file__)
if getattr(sys, "frozen", False):
    ROOT_PATH = os.path.dirname(sys.executable)

_turboshuf = os.path.join(ROOT_PATH, "turboshuf")


def turboshuf(filename, output, seed=None, skip=0, memory=4.0):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"file not found: {filename}")

    full = os.path.abspath(filename)
    parent_path = os.path.dirname(full)
    name, ext = os.path.splitext(os.path.basename(full))

    tmp_dir = os.path.join(parent_path, f"{name}-tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    env = {
        "SKIP": skip,
        "TMPDIR": tmp_dir,
        "MEMORY": memory,
        "SEED": seed,
    }
    envs = " ".join([f"{k}={v}" for k, v in env.items() if v is not None])

    cmd = f"{envs} {_turboshuf} <{filename} >{output}"
    try:
        check_output(cmd, shell=True, text=True)
    finally:
        os.rmdir(tmp_dir)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="shuf data fast")
    parser.add_argument("--filename", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--memory", type=float, default=4.0)

    args = parser.parse_args()

    turboshuf(filename=args.filename, output=args.output, seed=args.seed, skip=args.skip, memory=args.memory)
