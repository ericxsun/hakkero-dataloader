#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import itertools
import math
import random
import os

import numpy as np
from scipy.stats import qmc
from PIL import Image
from copy import deepcopy
from io import BytesIO
from PIL.Image import Image as ImageObject
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union
# Specifies a target value that is ignored and does not contribute to the input gradient
# see torch.nn.functional.cross_entropy
class EncodedImage(TypedDict):
    path: Optional[str]
    bytes: Optional[bytes]
ImageInput = Union[str, EncodedImage, ImageObject]
IMAGE_PLACEHOLDER = "<image>"
qwen2_image_token="<|image_pad|>"
IGNORE_INDEX = -100


class MultinomialSampler:
    # quasi monte-carlo sampler for better coherence to the expected categorical distribution
    def __init__(self, seed=-1):
        if seed < 0:
            self.sampler = itertools.count()
        else:
            self.sampler = qmc.Sobol(d=1, seed=seed)

    def next(self, weights):
        weight_sum = weights.sum()
        if np.isclose(weight_sum, 0):
            raise StopIteration()

        if isinstance(self.sampler, itertools.count):
            while True:
                i = next(self.sampler) % len(weights)
                if weights[i] > 0:
                    return i
        else:
            p_vals = weights / weight_sum

            residual = 1 - p_vals.sum()
            if residual > 0:
                p_vals[p_vals.argmin()] += residual
            else:
                p_vals[p_vals.argmax()] += residual

            return qmc.MultinomialQMC(p_vals, 1, engine=self.sampler).random(1).argmax()


def random_range(start, stop=None, step=None, seed=0):
    """Generator of non-repeated random permutation with the same interface of python `range`.

    Ref: https://stackoverflow.com/q/53551417

    The random.shuffle(list) and random.sample(list, len(list)) require materialize the lists, which result in a
    long initialization period.
    """
    if stop is None:
        start, stop = 0, start

    if step is None:
        step = 1

    # use a mapping to convert a standard range into the desired range
    mapping = lambda i: (i * step) + start

    # compute the number of numbers in this range
    maximum = int(math.ceil((stop - start) / step))

    # early return with empty range
    if maximum == 0:
        yield from ()
        return

    # seed range with a random integer
    value = random.randint(0, maximum)

    # Construct an offset, multiplier, and modulus for a linear
    # congruential generator. These generators are cyclic and
    # non-repeating when they maintain the properties:
    #
    #   1) "modulus" and "offset" are relatively prime.
    #   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    #   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4.
    #
    # Pick a random odd-valued offset.
    offset = random.randint(0, maximum) * 2 + 1
    # Pick a multiplier 1 greater than a multiple of 4.
    multiplier = 4 * (maximum // 4) + 1
    # Pick a modulus just big enough to generate all numbers (power of 2).
    modulus = int(2 ** math.ceil(math.log2(maximum)))
    # Track how many random numbers have been returned.
    found = 0
    while found < maximum:
        # If this is a valid value, yield it in generator fashion.
        if value < maximum:
            found += 1
            yield mapping(value)
        # Calculate the next value in the sequence.
        value = (value * multiplier + offset) % modulus


class Range:
    def __init__(self, start, stop, step):
        self.start, self.stop, self.step = start, stop, step

    def __repr__(self):
        return f"Range({self.start}, {self.stop}, {self.step})"

    def iterate(self):
        yield from range(self.start, self.stop, self.step)

    def list(self):
        return list(range(self.start, self.stop, self.step))

    def sub_range(self, split, n_splits):
        # strided split of range
        # e.g., [0, 3, 5, 7, 9] can be split into [0, 5, 9] and [3, 7]
        return Range(self.start + self.step * split, self.stop, self.step * n_splits)

    def random_iterate(self):
        yield from random_range(self.start, self.stop, self.step)

    def __len__(self):
        return math.ceil((self.stop - self.start) / self.step)


class RunningAverage:
    def __init__(self, default=100):
        self._default = default
        self._value = 0
        self._count = 0

    @property
    def value(self):
        if self._count == 0:
            return self._default

        return self._value

    def reset(self):
        self._value = 0
        self._count = 0

    def update(self, x):
        if isinstance(x, (list, tuple)):
            if x:
                self._count += len(x)
                xsum = sum(x, start=0.0)
                self._value += (xsum - len(x) * self._value) / self._count
        else:
            self._count += 1
            self._value += (x - self._value) / self._count

    def __repr__(self):
        return str(self.value)


def format_size(size_bytes):
    if size_bytes == 0:
        return "0B"

    names = ("B", "KB", "MB", "GB", "TB", "PB")

    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s}{names[i]}"


##---------MMLLM-----------
def _preprocess_image(image: "ImageObject",  **kwargs) -> "ImageObject":
        image_resolution: int = kwargs.get("image_resolution")
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

def _regularize_images(images,  **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError("Expect input is a list of Images, but got {}.".format(type(image)))

            results.append(_preprocess_image(image,  **kwargs))

        return results

def _get_mm_inputs(images, processor):
    # image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    # video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
    image_processor = getattr(processor, "image_processor")
    video_processor = getattr(processor, "video_processor", image_processor)
    input_dict = {"images": None}  # default key
    if len(images) != 0:
        images = _regularize_images(
            images,
            image_resolution=getattr(processor, "image_resolution", 512),
            )
        input_dict["images"] = images

    mm_inputs = {}
    if input_dict.get("images") is not None:
        mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))

    if input_dict.get("videos") is not None:
        mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))

    return mm_inputs

def message_trans(messages, base_path):
    res_messages = []
    images = []
    for i, message in enumerate(messages):
        role = message["role"]
        contents = message["content"]
        txt = ""
        for content in contents:
            if content["type"] == "image":
                img_path = os.path.join(base_path, content["image"])
                images.append(img_path)
                txt += "<image>"
            elif content["type"] == "text":
                txt += content["text"]
            else:
                NotImplemented
        res_messages.append(
            {
                "role": role,
                "content": txt
            }
            )
    return res_messages, images

def process_messages(
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        processor,
    ) -> List[Dict[str, str]]:
        
        image_processor = getattr(processor, "image_processor")
        merge_length = getattr(image_processor, "merge_size") ** 2
        mm_inputs = _get_mm_inputs(images, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])

        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError("`len(images)` is less than the number of {} tokens.".format(IMAGE_PLACEHOLDER))

                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        qwen2_image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))

        return messages, mm_inputs