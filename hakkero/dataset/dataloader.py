#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import itertools
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

from hakkero.dataset.utils import IGNORE_INDEX


def select(seq, indices, reverse=False):
    if not isinstance(indices, set):
        indices = set(indices)

    if reverse:
        return [seq for i, seq in enumerate(seq) if i not in indices]

    return [seq for i, seq in enumerate(seq) if i in indices]


class CudaPrefetcher(Iterable):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.next = None

        self.preload()

    def preload(self):
        try:
            self.next = next(self.loader)
        except StopIteration:
            self.next = None
            return

        with torch.cuda.stream(self.stream):
            for key in self.next.keys():
                if isinstance(self.next[key], torch.Tensor):
                    self.next[key] = self.next[key].cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        current = self.next

        if current is None:
            raise StopIteration()

        for value in current.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(torch.cuda.current_stream())

        self.preload()

        return current

    def __iter__(self):
        return self


class Loader(torch.utils.data.IterableDataset):
    def prefetch(self, num_workers=2, prefetch_factor=20, pin_memory=True, drop_last=False):
        loader = torch.utils.data.DataLoader(
            self,
            batch_size=None,
            collate_fn=lambda x: x,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return CudaPrefetcher(loader)

    def get_stats(self, task_ids, useds, failed):
        # task -> epoch -> set(indices)
        stats = dict()
        for task, used in itertools.chain(zip(task_ids, useds), failed):
            if task not in stats:
                stats[task] = defaultdict(set)
            for epoch, index in used:
                stats[task][epoch].add(index)
        return stats


class PadLoaderBase(Loader):
    def __init__(self, dataset, batch_size, padding_id, unpad=False):
        assert padding_id is not None, "padding_id should not be None"

        self.dataset = dataset
        self.padding_id = padding_id
        self.batch_size = batch_size
        self.unpad = unpad

        self.useds = []
        self.failed = []

        self.input_ids = []
        self.labels = []

        self.pixel_values = []
        self.image_grid_thw = []

        self.lengths = []
        self.n_targets = []
        self.task_ids = []

    def exceed(self):
        return len(self.labels) == self.batch_size

    def __iter__(self):
        for sample in self.dataset:
            self.put(sample)

            if not self.exceed():
                continue
            yield self.pop()

        if len(self.labels):
            yield self.pop()

    def put(self, sample):
        raise NotImplementedError()

    def pop(self):
        raise NotImplementedError()


class PadLoader(PadLoaderBase):
    def put(self, sample):
        if "input" not in sample:
            self.failed.append((sample["task"], sample["used"]))
            return

        self.useds.append(sample["used"])
        self.task_ids.append(sample["task"])

        self.input_ids.append(sample["input"])
        self.labels.append(sample["label"])
        if "pixel_values" in sample:
            self.pixel_values.extend(sample["pixel_values"])
        if "image_grid_thw" in sample:
            self.image_grid_thw.extend(sample["image_grid_thw"])

        self.lengths.append(sample["length"])
        self.n_targets.append(sample["n_target"])

    def pop(self):
        batch = {
            "n_samples": torch.tensor(len(self.lengths), dtype=torch.long),
            "n_tokens": torch.tensor(sum(self.lengths), dtype=torch.long),
            "n_targets": torch.tensor(sum(self.n_targets), dtype=torch.long),
            "stats": self.get_stats(self.task_ids, self.useds, self.failed),
        }

        if not self.unpad:
            batch["input_ids"] = pad_sequence(self.input_ids, batch_first=True, padding_value=self.padding_id).long()
            batch["labels"] = pad_sequence(self.labels, batch_first=True, padding_value=IGNORE_INDEX).long()
            batch["attention_mask"] = batch["input_ids"].ne(self.padding_id).long()
            batch["task_ids"] = torch.stack(
                [torch.full((batch["input_ids"].shape[-1],), task, dtype=torch.long) for task in self.task_ids], dim=0
            )
            if self.pixel_values:
                batch["pixel_values"] = torch.stack(self.pixel_values, dim=0)
            if self.image_grid_thw:
                batch["image_grid_thw"] = torch.stack(self.image_grid_thw, dim=0)
        else:
            batch["input_ids"] = torch.cat(self.input_ids, dim=0).unsqueeze(0)
            batch["labels"] = torch.cat(self.labels, dim=0).unsqueeze(0)
            batch["cu_seqlens"] = torch.tensor([0] + self.lengths).cumsum(dim=-1).int()
            batch["position_ids"] = torch.cat(
                [torch.arange(length, dtype=torch.long) for length in self.lengths], 0
            ).unsqueeze(0)
            batch["task_ids"] = torch.cat(
                [torch.full((l,), t, dtype=torch.long) for t, l in zip(self.task_ids, self.lengths)], dim=0
            )
            if self.pixel_values:
                batch["pixel_values"] = torch.cat(self.pixel_values, dim=0)
            if self.image_grid_thw:
                batch["image_grid_thw"] = torch.cat(self.image_grid_thw, dim=0)

        self.failed = []
        self.useds = []
        self.task_ids = []
        self.input_ids = []
        self.labels = []
        self.pixel_values = []
        self.image_grid_thw = []
        self.lengths = []
        self.n_targets = []

        return batch


class PreferencePadLoader(PadLoaderBase):
    def put(self, sample):
        if "inputs" not in sample:
            self.failed.append((sample["task"], sample["used"]))
            return

        self.useds.append(sample["used"])
        self.task_ids.append(sample["task"])

        self.input_ids.append(sample["inputs"])
        self.labels.append(sample["labels"])
        self.lengths.append(sample["length"])
        self.n_targets.append(sample["n_target"])

    def pop(self):
        batch = {
            "input_ids": dict(),
            "labels": dict(),
            "task_ids": dict(),
            "n_samples": torch.tensor(len(self.lengths), dtype=torch.long),
            "n_tokens": torch.tensor(
                sum(pl for part_lengths in self.lengths for pl in part_lengths.values()), dtype=torch.long
            ),
            "n_targets": torch.tensor(
                sum(pt for part_n_targets in self.lengths for pt in part_n_targets.values()), dtype=torch.long
            ),
            "stats": self.get_stats(self.task_ids, self.useds, self.failed),
        }

        if not self.unpad:
            batch["attention_mask"] = dict()

            for key in ("chosen", "rejected"):
                batch["input_ids"][key] = pad_sequence(
                    [d[key] for d in self.input_ids], batch_first=True, padding_value=self.padding_id
                ).long()
                batch["labels"][key] = pad_sequence(
                    [d[key] for d in self.labels], batch_first=True, padding_value=IGNORE_INDEX
                ).long()

                batch["attention_mask"][key] = batch["input_ids"][key].ne(self.padding_id).long()

                batch_length = batch["input_ids"][key].shape[-1]
                batch["task_ids"][key] = torch.stack(
                    [torch.full((batch_length,), task, dtype=torch.long) for task in self.task_ids], dim=0
                )
        else:
            batch["cu_seqlens"] = dict()
            batch["max_seqlen"] = dict()
            batch["position_ids"] = dict()

            for key in ("chosen", "rejected"):
                batch["input_ids"][key] = torch.cat([d[key] for d in self.input_ids], 0).unsqueeze(0)
                batch["labels"][key] = torch.cat([d[key] for d in self.labels], 0).unsqueeze(0)

                seqlens = [p[key] for p in self.lengths]
                batch["cu_seqlens"][key] = torch.tensor([0] + seqlens).cumsum(dim=-1).int()

                batch["max_seqlen"][key] = max(seqlens)

                batch["position_ids"][key] = torch.cat(
                    [torch.arange(l, dtype=torch.long) for l in seqlens], 0
                ).unsqueeze(0)

                batch["task_ids"][key] = torch.cat(
                    [torch.full((l,), t, dtype=torch.long) for t, l in zip(self.task_ids, seqlens)], 0
                ).unsqueeze(0)

        self.failed = []
        self.useds = []
        self.input_ids = []
        self.labels = []
        self.lengths = []
        self.n_targets = []
        self.task_ids = []
        return batch


class UnpadLoaderBase(Loader):
    def __init__(self, dataset, max_total_length, n_cache=3, max_explore=32):
        self.dataset = dataset
        self.max_total_length = max_total_length
        self.n_cache = n_cache
        self.max_explore = max_explore  # maximum comparison times for find_batch_combination
        self.total_length = 0

        self.useds = []
        self.failed = []

        self.lengths = []
        self.input_ids = []
        self.labels = []

        self.n_targets = []
        self.task_ids = []

        self.seqlens = []

    def find_batch_combination(self):
        # find the longest batch combination that is smaller than max_total_length
        if not self.lengths:
            return []

        if min(self.lengths) > self.max_total_length:
            raise ValueError(
                f"No the minimal segment length {min(self.lengths)} does not fit into max_total_length {self.max_total_length}"
            )

        # simple return all samples
        if self.total_length < self.max_total_length:
            return list(range(len(self.lengths)))

        # arrange the indices for better backtrack heuristics:
        # 1. fill the batch with random samples
        # 2. modify the batch by swapping the shorter samples
        # the batch is not necessarily biased as we are swapping shorter samples with shorter ones.
        i = 0
        for i, length_sum in enumerate(itertools.accumulate(self.lengths)):
            if length_sum > self.max_total_length:
                break

        bt_indices = np.argsort(self.lengths[:i])[::-1].tolist() + (np.argsort(self.lengths[i:]) + i).tolist()

        num_explore = 0
        best_indices = []
        best_total_length = 0
        stack = [(0, 0, [])]

        # depth-first backtracking with early stopping
        # the default combination will always be considered first
        # the try to modify the last elements of the combination
        while stack:
            bt_index, current_total_length, current_batch = stack.pop()
            if current_total_length > self.max_total_length:
                # skip as future combination will certainly exceed max_total_length
                continue
            elif current_total_length > best_total_length:
                # improved
                best_total_length = current_total_length
                best_indices = current_batch[:]
            else:
                # fail to improve
                num_explore += 1

            # early stop to avoid too much comparison
            if num_explore > self.max_explore:
                break

            # stop if we've considered all samples
            if bt_index == len(self.lengths):
                break

            # exclude the current sample and explore further
            stack.append((bt_index + 1, current_total_length, current_batch))

            # include the current sample and explore further
            stack.append(
                (
                    bt_index + 1,
                    current_total_length + self.lengths[bt_indices[bt_index]],
                    current_batch + [bt_indices[bt_index]],
                )
            )

        return best_indices

    def __iter__(self):
        for sample in self.dataset:
            self.put(sample)
            if self.exceed():
                yield self.pop()
        while self.input_ids:
            yield self.pop()

    def exceed(self):
        return self.total_length > self.n_cache * self.max_total_length

    def put(self, sample):
        raise NotImplementedError()

    def pop(self):
        raise NotImplementedError()


class UnpadLoader(UnpadLoaderBase):
    def put(self, sample):
        if "input" not in sample:
            self.failed.append((sample["task"], sample["used"]))
            return

        self.task_ids.append(sample["task"])
        self.useds.append(sample["used"])

        self.lengths.append(sample["length"])
        self.n_targets.append(sample["n_target"])
        self.input_ids.append(sample["input"])
        self.labels.append(sample["label"])
        self.total_length += sample["length"]

    def pop(self):
        indices = self.find_batch_combination()
        lengths = select(self.lengths, indices)
        n_targets = select(self.n_targets, indices)
        task_ids = select(self.task_ids, indices)
        useds = select(self.useds, indices)

        input_ids = select(self.input_ids, indices)
        labels = select(self.labels, indices)

        batch = {
            "input_ids": torch.cat(input_ids, 0).unsqueeze(0),
            "labels": torch.cat(labels, 0).unsqueeze(0),
            "n_samples": torch.tensor(len(lengths), dtype=torch.long),
            "n_tokens": torch.tensor(sum(lengths), dtype=torch.long),
            "n_targets": torch.tensor(sum(n_targets), dtype=torch.long),
            "stats": self.get_stats(task_ids, useds, self.failed),
            "task_ids": torch.cat(
                [torch.full((l,), t, dtype=torch.long) for t, l in zip(task_ids, lengths)], dim=0
            ).unsqueeze(0),
            "position_ids": torch.cat([torch.arange(length, dtype=torch.long) for length in lengths], 0).unsqueeze(0),
            "cu_seqlens": torch.tensor([0] + lengths).cumsum(dim=-1).int(),
            "max_seqlen": max(lengths),
        }

        self.lengths = select(self.lengths, indices, reverse=True)
        self.n_targets = select(self.n_targets, indices, reverse=True)
        self.task_ids = select(self.task_ids, indices, reverse=True)
        self.useds = select(self.useds, indices, reverse=True)
        self.total_length = sum(self.lengths)

        self.input_ids = select(self.input_ids, indices, reverse=True)
        self.labels = select(self.labels, indices, reverse=True)

        self.failed = []

        return batch


class PreferenceUnpadLoader(UnpadLoaderBase):
    def put(self, sample):
        if "inputs" not in sample:
            self.failed.append((sample["task"], sample["used"]))
            return

        length = sum(sample["length"].values())
        self.lengths.append(length)
        self.total_length += length
        n_target = sum(sample["n_target"].values())
        self.n_targets.append(n_target)

        self.task_ids.append(sample["task"])
        self.useds.append(sample["used"])
        self.input_ids.append(sample["inputs"])
        self.labels.append(sample["labels"])

        self.seqlens.append(sample["length"])

    def pop(self):
        indices = self.find_batch_combination()

        lengths = select(self.lengths, indices)
        n_targets = select(self.n_targets, indices)
        task_ids = select(self.task_ids, indices)
        useds = select(self.useds, indices)

        part_input_ids = select(self.input_ids, indices)
        part_labels = select(self.labels, indices)
        part_seqlens = select(self.seqlens, indices)

        # task -> epoch -> set(indices)
        stats = dict()
        for task, used in itertools.chain(zip(task_ids, useds), self.failed):
            if task not in stats:
                stats[task] = defaultdict(set)
            for epoch, index in used:
                stats[task][epoch].add(index)

        batch = {
            "input_ids": dict(),
            "labels": dict(),
            "task_ids": dict(),
            "sample_ids": dict(),
            "position_ids": dict(),
            "cu_seqlens": dict(),
            "max_seqlen": dict(),
            "n_samples": torch.tensor(len(lengths), dtype=torch.long),
            "n_tokens": torch.tensor(sum(lengths), dtype=torch.long),
            "n_targets": torch.tensor(sum(n_targets), dtype=torch.long),
            "stats": self.get_stats(self.task_ids, self.useds, self.failed),
        }

        for key in ("chosen", "rejected"):
            batch["input_ids"][key] = torch.cat([d[key] for d in part_input_ids], 0).unsqueeze(0)
            batch["labels"][key] = torch.cat([d[key] for d in part_labels], 0).unsqueeze(0)

            seqlens = [part_seqlen[key] for part_seqlen in part_seqlens]
            batch["task_ids"][key] = torch.cat(
                [torch.full((l,), t, dtype=torch.long) for t, l in zip(task_ids, seqlens)], 0
            ).unsqueeze(0)
            batch["sample_ids"][key] = torch.cat(
                [torch.full((l,), s, dtype=torch.long) for s, l in enumerate(seqlens)], 0
            ).unsqueeze(0)
            batch["position_ids"][key] = torch.cat([torch.arange(l, dtype=torch.long) for l in seqlens], 0).unsqueeze(0)
            batch["cu_seqlens"][key] = (torch.tensor([0] + seqlens).cumsum(dim=-1).int(),)
            batch["max_seqlen"][key] = max(seqlens)

        self.lengths = select(self.lengths, indices, reverse=True)
        self.n_targets = select(self.n_targets, indices, reverse=True)
        self.task_ids = select(self.task_ids, indices, reverse=True)
        self.useds = select(self.useds, indices, reverse=True)
        self.total_length = sum(self.lengths)

        self.input_ids = select(self.input_ids, indices, reverse=True)
        self.labels = select(self.labels, indices, reverse=True)
        self.seqlens = select(self.seqlens, indices, reverse=True)

        self.failed = []
        return batch
