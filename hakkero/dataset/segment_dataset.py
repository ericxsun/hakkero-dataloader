#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import os.path
import random

import numpy as np
import torch.utils.data

from hakkero.dataset.iterable_dataset import IterableDataset
from hakkero.dataset.logger import logger
from hakkero.dataset.strategy import default_strategy
from hakkero.dataset.strategy import segment as strategy_segment
from hakkero.dataset.strategy import SegmentationError
from hakkero.dataset.strategy import ST_CONCAT
from hakkero.dataset.strategy import ST_SEGMENT
from hakkero.dataset.strategy import ST_TOKENIZE
from hakkero.dataset.strategy import TokenizationError
from hakkero.dataset.strategy import tokenize as strategy_tokenize
from hakkero.dataset.utils import MultinomialSampler
from hakkero.dataset.utils import RunningAverage


class SegmentDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        path,
        name=None,
        tokenizer=None,
        prefetcher=None,
        seed=-1,
        infinite=False,
        max_epoch=1,
        strategy=None,
        max_length=1024,
        n_shards=2,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super().__init__()

        self.path = path
        self.name = name if name else os.path.basename(path)
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        block_size = kwargs.pop("block_size", 1024)
        self.dataset = IterableDataset(
            path,
            name=name,
            max_epoch=max_epoch,
            n_shards=n_shards,
            seed=seed,
            block_size=block_size,
            prefetcher=prefetcher,
            infinite=infinite,
            rank=rank,
            world_size=world_size,
        )

        self.strategy = default_strategy if strategy is None else strategy

        self.tokenizer = tokenizer
        self.processor = kwargs.pop("processor", None)
        self.max_length = max_length

        self.prev = None
        self.length = RunningAverage()
        self.n_target = RunningAverage()
        self.random = None if self.seed < 0 else random.Random(self.seed)

        self.kwargs = kwargs

    def __next__(self):
        sample = next(self.dataset)
        if "data" not in sample:
            return [dict(used=[sample["info"]])]

        try:
            data = strategy_tokenize[self.strategy[ST_TOKENIZE]](
                sample["data"], self.tokenizer, processor=self.processor, path=self.path, **self.kwargs
            )
        except TokenizationError as e:
            logger.warning(f"[rank={self.rank}, world_size={self.world_size}][{self.path}:{sample['info'][1]}]: {e} => {sample['uid']}\n{sample['data']}")
            return [dict(used=[sample["info"]])]

        try:
            if self.strategy[ST_SEGMENT] == ST_CONCAT:
                while True:
                    segments, self.prev = strategy_segment[self.strategy[ST_SEGMENT]](
                        data, self.max_length, sample["info"], self.random, self.prev
                    )
                    if segments:
                        break
            else:
                segments = strategy_segment[self.strategy[ST_SEGMENT]](
                    data, self.max_length, sample["info"], self.random
                )
        except SegmentationError as e:
            logger.warning(f"[rank={self.rank}, world_size={self.world_size}][{self.path}:{sample['info'][1]}]: {e} => {sample['uid']}")
            return [dict(used=[sample["info"]])]

        for s in segments:
            if "labels" in s:
                # preference data
                assert "inputs" in s
                s["length"] = {key: value.nelement() for key, value in s["labels"].items()}
                s["n_target"] = {key: value.gt(0).count_nonzero().item() for key, value in s["labels"].items()}
                self.length.update(list(s["length"].values()))
                self.n_target.update(list(s["n_target"].values()))
            else:
                # pretrain or sft
                s["length"] = s["label"].nelement()
                s["n_target"] = s["label"].gt(0).count_nonzero().item()
                self.length.update(s["length"])
                self.n_target.update(s["n_target"])

        return segments

    def __iter__(self):
        iter(self.dataset)
        self.prev = None
        self.length = RunningAverage()
        self.n_target = RunningAverage()
        self.random = None if self.seed < 0 else random.Random(self.seed)
        return self

    @property
    def avg_length(self):
        assert self.length.value > 0, f"{self.path} has zero length, please ensure the dataset is well-formed"
        return self.length.value

    @property
    def avg_n_target(self):
        assert self.n_target.value > 0, f"{self.path} has zero target tokens, please ensure the dataset is well-formed"
        return self.n_target.value

    def state_dict(self):
        # {"size": size, "epoch": epoch, "bsize": block_size, "states": {epoch: BlockedBitMap()}}
        return self.dataset.state_dict()

    def load_state_dict(self, state_or_state_dict):
        self.dataset.load_state_dict(state_or_state_dict)

    def track(self, stats):
        # states: {epoch: indices}
        self.dataset.track(stats)


class MixedSegmentDataset(SegmentDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # fewer slots for less mixed distribution but faster initial loading
        self.n_slots = 1 if self.seed < 0 else 20
        self.weights = np.array([1 / self.n_slots] * self.n_slots)
        self.sampler = MultinomialSampler(self.seed)
        self.slots = [[] for _ in range(self.n_slots)]
        self.remain = None
        self.exhausted = False

    @property
    def size(self):
        return self.dataset.size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        super().__iter__()
        self.weights = np.array([1 / self.n_slots] * self.n_slots, dtype=np.float64)
        self.sampler = MultinomialSampler(self.seed if self.seed >= 0 else 0)
        self.slots = [[] for _ in range(self.n_slots)]
        self.remain = None
        self.exhausted = False

        return self

    def __next__(self):
        while True:
            # select a non-empty slot
            try:
                i = self.sampler.next(self.weights)
            except StopIteration:
                self.exhausted = True
                raise StopIteration()

            if not self.slots[i]:
                try:
                    # try to fill the slot
                    self.slots[i] = super().__next__()
                except StopIteration:
                    # no new segments, mask the slot and move on
                    self.weights[i] = 0
                    continue

            # the current slot is selected and filled
            break

        # maybe start count down for the infinite case
        if self.dataset.exhausted:
            if self.remain is None:
                self.remain = sum(len(slot) for slot in self.slots)

            self.remain -= 1

            if self.remain <= 0:
                self.exhausted = True

        return self.slots[i].pop(0)
