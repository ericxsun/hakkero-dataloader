#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import itertools
import random

import torch

from hakkero.dataset.strategy.errors import SegmentationError


def integrous(data, max_length, info, r: random.Random = None):
    """normal way: discard sample that is too long, exceed max_length"""
    # pretrain & sft

    if "pixel_values" in data and "image_grid_thw" in data:
        length = data["input"].nelement()
        assert length == data["label"].nelement()
        if length > max_length:
            raise SegmentationError(f"input length {length} > {max_length}")

        return [
            dict(
                input=data["input"],
                label=data["label"],
                pixel_values=data["pixel_values"],
                image_grid_thw=data["image_grid_thw"],
                used=[info],
            )
        ]

    if "input" in data:
        length = data["input"].nelement()
        assert length == data["label"].nelement()
        if length > max_length:
            raise SegmentationError(f"input length {length} > {max_length}")

        return [dict(input=data["input"], label=data["label"], used=[info])]

    # preference data
    if "inputs" in data:
        for key in data["inputs"].keys():
            length = data["inputs"][key].nelement()
            assert length == data["labels"][key].nelement()
            if length > max_length:
                raise SegmentationError(f"length {length} of `{key} > {max_length}")

        return [dict(inputs=data["inputs"], labels=data["labels"], used=[info])]

    raise ValueError("not supported data: {data}")


def concat(data, max_length, info, r: random.Random = None, prev=None):
    """split long and concat it with previous segment"""
    if "input" not in data and "inputs" in data:
        raise ValueError("preference data only support integrous segmentation")

    if prev is not None:
        # concat input and label with previous segment
        input = torch.cat([prev["input"], prev["label"][-1:], data["input"]])
        label = torch.cat([prev["label"], data["input"][0:1], data["label"]])
        first_used = prev["used"] + [info]
        later_used = [info]
    else:
        input = data["input"]
        label = data["label"]

        first_used = later_used = [info]

    length = input.nelement()
    if length > max_length:
        segments = [
            dict(
                input=input[st : st + max_length],
                label=label[st : st + max_length],
                used=first_used if st == 0 else later_used,
            )
            for st in range(0, length, max_length)
        ]
    else:
        segments = [dict(input=input, label=label, used=first_used)]

    if segments[-1]["input"].nelement() == max_length:
        prev = None
    else:
        # concat the last segment to the next sample
        prev = segments.pop()

    if segments and r is not None:
        r.shuffle(segments)

    return segments, prev


def naive(data, max_length, info, r: random.Random = None):
    if "input" not in data and "inputs" in data:
        raise ValueError("preference data only support integrous segmentation")

    used = [info]

    length = data["input"].nelement()
    if length <= max_length:
        return [dict(input=data["input"], label=data["label"], used=used)]

    segments = [
        dict(input=data["input"][st:ed], label=data["label"][st:ed], used=used)
        for st, ed in map(lambda i: (i, min(i + max_length, length)), range(0, length, max_length))
    ]

    if r is not None:
        r.shuffle(segments)

    return segments


def unbiased(data, max_length, info, r: random.Random = None):
    if "input" not in data and "inputs" in data:
        raise ValueError("preference data only support integrous segmentation")

    assert r is not None

    used = [info]

    length = data["input"].nelement()
    if length <= max_length:
        return [dict(input=data["input"], label=data["label"], used=used)]

    segment_lens = [max_length] * (length // max_length)
    remain = length % max_length
    if remain != 0:
        shift = r.randint(0, max_length - remain)
        segment_lens.append(segment_lens.pop() - shift)
        segment_lens.append(remain + shift)
        # ensure the short segments can appear at any position of the long sequence
        r.shuffle(segment_lens)

    bounds = [0] + list(itertools.accumulate(segment_lens))

    segments = [
        dict(input=data["input"][st:ed], label=data["label"][st:ed], used=used)
        for st, ed in zip(bounds[:-1], bounds[1:])
    ]

    r.shuffle(segments)

    return segments
