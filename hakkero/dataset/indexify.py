#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from functools import partial

import h5py

from hakkero import __version__
from hakkero.dataset.logger import logger
from hakkero.dataset.strategy.check import check_func


def _build_chunk_offsets(filename, start, end, worker_id, n_workers, dtype="legacy"):
    offset = start
    bounds = []
    with open(filename, "rb") as fin:
        fin.seek(offset)

        if start > 0:
            fin.readline()
            offset = fin.tell()

        bounds.append(offset)

        while offset < end:
            logger.info(f"worker: {worker_id}/{n_workers}, offset: {offset}, range in offset [{start}, {end}]")
            line = fin.readline()
            offset += len(line)
            bounds.append(offset)

            try:
                js = json.loads(line)
                _ = js["uid"]
                data = js["data"]

                status, msg = check_func[dtype](data)
                if status:
                    continue
                else:
                    raise ValueError(f"\ndata: {line}\nerror: {msg}")
            except Exception as e:
                logger.error(f"invalid format line: {line}\n{e}")
                raise e

    return bounds


def build_index(filename, output=None, num_workers=None, dtype="legacy", shuf=True):
    if dtype not in list(check_func.keys()):
        raise ValueError(f"dtype must be one of {list(check_func.keys())}")

    if shuf:
        from hakkero.lib import turboshuf

        logger.info("shuf before building index")
        st = time.time()
        try:
            logger.info(f"[shuf] mv {filename} => {filename}.tmp")
            shutil.move(filename, f"{filename}.tmp")

            logger.info(f"[shuf] shuf {filename}.tmp => {filename}")
            turboshuf(f"{filename}.tmp", filename)

            logger.info(f"[shuf] rm {filename}.tmp")
            os.remove(f"{filename}.tmp")
        except Exception as e:
            logger.error(f"[shuf] shuf failed: {e}\nfail back...not shuf...")
            try:
                shutil.move(f"{filename}.tmp", filename)
            except Exception:
                pass

        es = time.time()
        logger.info(f"[shuf] shuf cost: {es - st}(sec)")

    if num_workers is None:
        num_workers = mp.cpu_count()

    logger.info(f"build indexed with {num_workers} workers for dataset from {filename}")

    file_size = os.path.getsize(filename)
    chunk_size = file_size // num_workers

    pool = mp.Pool(processes=num_workers)

    chunks = [(i * chunk_size, (i + 1) * chunk_size, i + 1, num_workers, dtype) for i in range(num_workers)]
    chunks[-1] = (chunks[-1][0], file_size, chunks[-1][2], num_workers, dtype)

    func = partial(_build_chunk_offsets, filename)
    results = pool.starmap(func, chunks)

    pool.close()
    pool.join()

    bounds = [0]
    for chunk_offsets in results:
        if bounds[-1] != chunk_offsets[0]:
            bounds.append(chunk_offsets[0])
        bounds.extend(chunk_offsets[1:])

    bounds = sorted(bounds)
    if bounds[-1] != file_size:
        bounds.append(file_size)

    if output is not None:
        os.makedirs(output, exist_ok=True)
    else:
        output = os.path.dirname(filename)

    logger.info(f"build index.h5 into {output}")
    with h5py.File(os.path.join(output, "index.h5"), "w") as hf:
        hf.create_dataset("index", data=bounds)

    logger.info(f"build data.jsonl into {output}")
    if not os.path.exists(os.path.join(output, "data.jsonl")):
        shutil.copyfile(filename, os.path.join(output, "data.jsonl"))


def main():
    parser = argparse.ArgumentParser(description="build index for dataset")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--filename", type=str, help="full filename of jsonl file")
    parser.add_argument("--output", type=str, help="output path for saving data.jsonl and index.h5")
    parser.add_argument("--dtype", type=str, choices=list(check_func.keys()), help="data type", required=True)
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers")
    parser.add_argument("--not_shuf", action="store_true", help="not shuf data")

    args = parser.parse_args()

    build_index(args.filename, args.output, args.num_workers, args.dtype, not args.not_shuf)


if __name__ == "__main__":
    main()
