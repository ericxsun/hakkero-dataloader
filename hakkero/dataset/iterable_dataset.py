#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import itertools
import math
import os.path
import queue
import random
import threading
import time
from collections import defaultdict

import torch
import torch.utils.data
from bitarray import bitarray
from bitarray.util import deserialize
from bitarray.util import serialize
from tabulate import tabulate

from hakkero.dataset.indexed_dataset import IndexedDataset
from hakkero.dataset.logger import logger
from hakkero.dataset.utils import Range


class CacheEmpty(Exception):
    pass


class CacheFull(Exception):
    pass


class BlockedBitMap:
    def __init__(self, size, bsize):
        assert bsize & (bsize - 1) == 0

        self.size = size
        self.bsize = bsize
        self.last_bid = int(math.ceil(size / bsize)) - 1
        self.last_bsize = size % bsize if size % bsize else bsize
        self.obsolete = bitarray(int(math.ceil(size / bsize)))
        self.active = dict()

    def __repr__(self):
        s = f"obsolete({self.obsolete.count()}/{len(self.obsolete)}): {self.obsolete.to01()}"
        active = tabulate([[key, value.to01()] for key, value in self.active.items()], ["id", "bitarray"], "plain")

        if self.active:
            s += f"\nactive({len(self.active)}):\n{active}"

        return s

    def state(self):
        return serialize(self.obsolete), {k: serialize(v) for k, v in self.active.items()}

    @staticmethod
    def from_state(size, bsize, state):
        bitmap = BlockedBitMap(size, bsize)
        bitmap.obsolete = deserialize(state[0])
        bitmap.active = {k: deserialize(v) for k, v in state[1].items()}
        return bitmap

    def resize(self, bsize):
        if bsize == self.bsize:
            return

        assert bsize & (bsize - 1) == 0

        last_bid = int(math.ceil(self.size / bsize)) - 1
        last_bsize = self.size % bsize if self.size % bsize else bsize
        obsolete = bitarray(int(math.ceil(self.size / bsize)))
        active = dict()

        if self.bsize > bsize:
            ratio = self.bsize // bsize
            # expand obsolete bitarray
            for old_bid, used in enumerate(self.obsolete):
                if not used:
                    continue

                bid_st = old_bid * ratio
                bid_ed = min(bid_st + ratio, len(obsolete))
                obsolete[bid_st:bid_ed] = 1
            # split active bitarray and mark exhausted ones in obsolete
            for old_bid, abarray in self.active.items():
                for i, sid_st in enumerate(range(0, len(abarray), bsize)):
                    bid = old_bid * ratio + i
                    sid_ed = min(sid_st + bsize, len(abarray))
                    if abarray[sid_st:sid_ed].all():
                        obsolete[bid] = 1
                    else:
                        active[bid] = abarray[sid_st:sid_ed]
        else:
            ratio = bsize // self.bsize
            # concat active bitarrays
            for old_bid, abarray in self.active.items():
                bid = old_bid // ratio
                if bid not in active:
                    if bid == last_bid:
                        active[bid] = bitarray(last_bsize)
                    else:
                        active[bid] = bitarray(bsize)

                sid_st = old_bid % ratio * self.bsize
                sid_ed = min(sid_st + self.bsize, len(active[bid]))
                active[bid][sid_st:sid_ed] |= abarray

            # shrink obsolete bitarray, update active bitarrays
            for bid, old_bid_st in enumerate(range(0, len(self.obsolete), ratio)):
                old_bid_ed = min(old_bid_st + ratio, len(self.obsolete))
                if self.obsolete[old_bid_st:old_bid_ed].all():
                    # the merged block is obsolete
                    obsolete[bid] = 1
                else:
                    # the merged block is active, mark the used samples
                    if bid not in active:
                        if bid == last_bid:
                            active[bid] = bitarray(last_bsize)
                        else:
                            active[bid] = bitarray(bsize)

                    for i, used in enumerate(self.obsolete[old_bid_st:old_bid_ed]):
                        if not used:
                            continue

                        sid_st = i * self.bsize
                        sid_ed = min(sid_st + self.bsize, len(active[bid]))
                        active[bid][sid_st:sid_ed] = 1

        for bid in list(active.keys()):
            if not active[bid].any():
                del active[bid]

        self.bsize = bsize
        self.last_bid = last_bid
        self.last_bsize = last_bsize
        self.obsolete = obsolete
        self.active = active

    def set(self, indices):
        group = defaultdict(list)
        for i in indices:
            group[i // self.bsize].append(i % self.bsize)

        for bid, sids in group.items():
            if self.obsolete[bid]:
                continue

            if bid not in self.active:
                if bid == self.last_bid:
                    self.active[bid] = bitarray(self.last_bsize)
                else:
                    self.active[bid] = bitarray(self.bsize)

            try:
                self.active[bid][sids] = 1
            except Exception as e:
                raise e

            if self.active[bid].all():
                self.obsolete[bid] = 1
                del self.active[bid]

    def __eq__(self, other):
        if self.obsolete != other.obsolete:
            return False

        if self.active.keys() != other.active.keys():
            return False

        for bid, abarray in self.active.items():
            if other.active[bid] != abarray:
                return False

        return True

    def __iadd__(self, other):
        assert self.size == other.size
        assert self.bsize == other.bsize

        self.obsolete |= other.obsolete
        for bid, abarray in other.active.items():
            if bid in self.active:
                self.active[bid] |= abarray
                if self.active[bid].all():
                    self.obsolete[bid] = 1
                    del self.active[bid]
            else:
                self.active[bid] = abarray

        return self

    def block_used(self, bid):
        if self.obsolete[bid]:
            return True

    def sample_used(self, index):
        bid = index // self.bsize
        if self.obsolete[bid]:
            return True
        elif index is not None and bid in self.active and self.active[bid][index % self.bsize]:
            return True
        else:
            return False

    def all(self):
        return self.obsolete.all()


class IteratorState:
    def __init__(self, size, bsize):
        self.size = size
        self.bsize = bsize
        self.epoch = 0
        self.bitmaps = dict()  # {epoch: BlockedBitMap}

    def __repr__(self):
        s = f"IteratorState({self.size}, {self.bsize}) @ epoch {self.epoch}"
        return "\n".join([s] + [f"epoch {epoch}:\n{repr(bitmap)}" for epoch, bitmap in self.bitmaps.items()])

    def block_used(self, epoch, bid):
        if epoch < self.epoch:
            return True
        if epoch not in self.bitmaps:
            return False

        return self.bitmaps[epoch].block_used(bid)

    def sample_used(self, epoch, index):
        if epoch < self.epoch:
            return True
        if epoch not in self.bitmaps:
            return False

        return self.bitmaps[epoch].sample_used(index)

    def track(self, stats):
        # stats: {epoch: indices}
        for epoch, indices in stats.items():
            if epoch < self.epoch:
                continue
            if epoch not in self.bitmaps:
                self.bitmaps[epoch] = BlockedBitMap(self.size, self.bsize)
            self.bitmaps[epoch].set(indices)

            if self.bitmaps[epoch].all():
                self.epoch = epoch + 1
                del self.bitmaps[epoch]

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "size": self.size,
            "bsize": self.bsize,
            "bitmaps": {epoch: bitmap.state() for epoch, bitmap in self.bitmaps.items()},
        }

    def cleanup(self):
        # clean up exhausted bitmaps and update self.epoch
        epochs = sorted(self.bitmaps.keys(), reverse=True)
        for epoch in epochs:
            if self.bitmaps[epoch].all():
                # epoch start from the largest exhausted epoch
                self.epoch = epoch + 1
                break

        for epoch in epochs:
            # clean up exhausted bitmaps to save memory
            if epoch < self.epoch:
                del self.bitmaps[epoch]

    @staticmethod
    def from_state_dict(state_dict):
        # wrap_data = True: directly wrap the state_dict instead of copying to save space
        state = IteratorState(state_dict["size"], state_dict["bsize"])
        state.epoch = state_dict["epoch"]
        state.bitmaps = {
            k: BlockedBitMap.from_state(state.size, state.bsize, v) for k, v in state_dict["bitmaps"].items()
        }
        return state

    def reset(self):
        self.epoch = 0
        self.bitmaps = dict()

    def __iadd__(self, other):
        if self.bsize != other.bsize:
            other.resize(self.bsize)

        for epoch, bitmap in other.bitmaps.items():
            if epoch in self.bitmaps:
                self.bitmaps[epoch] += bitmap
            else:
                self.bitmaps[epoch] = bitmap

        return self

    def update(self, state_dict):
        state = IteratorState.from_state_dict(state_dict)
        self.__iadd__(state)


class IterableDataset(IndexedDataset, torch.utils.data.IterableDataset):
    def __init__(
        self,
        path,
        name=None,
        seed=-1,
        max_epoch=1,
        block_size=1024,
        prefetcher=None,
        infinite=False,
        n_shards=1,
        rank=0,
        world_size=1,
    ):
        super().__init__(path)

        self.name = name if name else os.path.basename(path)
        self.rank, self.world_size = rank, world_size
        self.n_shards = n_shards

        # infinite=False: stop immediately when current epoch exceeds self.max_epoch
        # infinite=True: simple mark self.exhausted and keep iterating
        self.infinite = infinite
        self.max_epoch = max_epoch
        self.seed = seed
        self.block_size, self.n_blocks = self.safe_block(block_size)

        self.lock = threading.RLock()
        self.cache = self.blocks = None  # lock protected variables
        self.prefetcher = prefetcher
        self.state = IteratorState(len(self), self.block_size)

        # intermediate states
        self.block_iter = None
        self.random = None
        self.exhausted = None

    def load_state_dict(self, state_or_state_dict):
        if isinstance(state_or_state_dict, IteratorState):
            self.state = state_or_state_dict
        else:
            self.state = IteratorState.from_state_dict(state_or_state_dict)

    def state_dict(self):
        return self.state.state_dict()

    def track(self, stats):
        self.state.track(stats)

    def safe_block(self, block_size):
        assert (block_size & (block_size - 1)) == 0, f"block_size should be power of 2, but got {block_size}."

        if len(self) < self.n_shards:
            raise ValueError(f"cannot split {len(self)} entries of {self.path} into {self.n_shards} shards")

        n_blocks = int(math.ceil(len(self) / block_size))
        if n_blocks < self.n_shards:
            block_size = len(self) // self.n_shards
            block_size = int(math.pow(2, math.floor(math.log(block_size) / math.log(2))))
            n_blocks = int(math.ceil(len(self) / block_size))

        return block_size, n_blocks

    @property
    def priority(self):
        return 2 - len(self.cache)

    def prefetch(self):
        with self.lock:
            if len(self.cache) == 2:
                raise CacheFull

            block = next(self.blocks)
            st = block[1] * self.block_size
            ed = min(st + self.block_size, len(self))

            bytes = self.get_bytes(slice(st, ed))
            samples = [dict(info=(block[0], i), byte=byte) for i, byte in zip(range(st, ed), bytes)]
            self.cache.append(samples)

    def build_next_iter(self, samples):
        if self.random is not None:
            self.random.shuffle(samples)

        for sample in samples:
            epoch, index = sample["info"]
            if self.state.sample_used(epoch, index):
                continue

            data = self.decode_byte(index, sample.pop("byte"))
            if data:
                sample["data"] = data["data"]
                sample["uid"] = data["uid"]

            yield sample

    def next(self, block=False):
        # handle transition across blocks
        # raise CacheNotFilled when cache is to be filled
        # when infinite = False: raise StopIteration when the dataset is exhausted
        # when infinite = True: silently mark exhausted = True
        while True:
            if self.exhausted and not self.infinite:
                raise StopIteration(f"Exhausted dataset '{self.path}'")

            try:
                return next(self.block_iter)
            except StopIteration:
                while not self.cache:
                    if block:
                        time.sleep(0.1)  # wait for the prefetcher to load next block
                    else:
                        raise CacheEmpty  # signal prefetcher to load next block

                samples = self.cache.pop(0)
                epoch = samples[0]["info"][0]
                if epoch >= self.max_epoch:  # epoch start from 0
                    self.exhausted = True

                self.block_iter = self.build_next_iter(samples)

    def __next__(self):
        if self.prefetcher is None:
            self.prefetcher = Prefetcher()

        return self.prefetcher.next(self)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            n_workers, worker_id = 1, 0
        else:
            n_workers, worker_id = worker_info.num_workers, worker_info.id

        r = Range(0, self.n_blocks, 1)
        r = r.sub_range(split=self.rank, n_splits=self.world_size)  # split index among multi-gpu workers
        r = r.sub_range(split=worker_id, n_splits=n_workers)  # split index among multiprocess dataloader workers

        logger.debug(
            f"world_size: {self.world_size}, rank: {self.rank}, n_workers: {n_workers}, worker_id: {worker_id}, r={r.list()}"
        )

        if self.seed >= 0:
            blocks = ((epoch, bid) for epoch in itertools.count(self.state.epoch) for bid in r.random_iterate())
            self.random = random.Random(self.seed)
        else:
            blocks = ((epoch, bid) for epoch in itertools.count(self.state.epoch) for bid in r.iterate())
            self.random = None

        blocks = itertools.filterfalse(lambda block: self.state.block_used(block[0], block[1]), blocks)

        self.exhausted = self.state.epoch >= self.max_epoch

        self.block_iter = iter(())
        with self.lock:
            self.cache = []
            self.blocks = blocks

        return self


class Prefetcher:
    # use a joint thread pool to prefetch multiple datasets random access of a list of PrefetchIterableDataset
    # prioritize prefetching dataset that is being consumed
    def __init__(self):
        self.datasets = []
        self.urgent = queue.Queue()
        self.should_stop = threading.Event()
        self.thread = None

    def stop_thread(self):
        if self.alive:
            self.should_stop.set()
            self.thread.join(timeout=60)

    @property
    def alive(self):
        return isinstance(self.thread, threading.Thread) and self.thread.is_alive()

    def __del__(self):
        self.stop_thread()

    def worker(self):
        # prefetch blocks of each dataset
        while not self.should_stop.is_set():
            try:
                dataset = self.urgent.get_nowait()  # prioritize urgent load
            except queue.Empty:
                if self.datasets:
                    # prioritize the least cached datasets
                    dataset = max(self.datasets, key=lambda d: d.priority)
                else:
                    time.sleep(0.001)
                    continue

            try:
                dataset.prefetch()
            except (CacheFull, StopIteration):
                time.sleep(0.001)

    def next(self, dataset):
        if not self.alive:
            self.thread = threading.Thread(target=self.worker, daemon=True)
            self.thread.start()

        if dataset not in self.datasets:
            self.datasets.append(dataset)

        try:
            return dataset.next(block=False)
        except CacheEmpty:
            self.urgent.put(dataset)
            return dataset.next(block=True)
