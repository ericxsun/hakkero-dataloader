#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import os
from copy import deepcopy
from io import BytesIO
from typing import List
from typing import Optional
from typing import TypedDict
from typing import Union

from PIL import Image
from PIL.Image import Image as ImageObject
from PIL.Image import Resampling


class EncodedImage(TypedDict):
    path: Optional[str]
    bytes: Optional[bytes]


ImageInput = Union[str, EncodedImage, ImageObject]
IMAGE_PLACEHOLDER = "<image>"


def preprocess(image: "ImageObject", **kwargs) -> "ImageObject":
    image_resolution: int = kwargs.get("image_resolution")

    if max(image.width, image.height) > image_resolution:
        resize_factor = image_resolution / max(image.width, image.height)
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")
    if min(image.width, image.height) < 28:
        width, height = max(image.width, 28), max(image.height, 28)
        image = image.resize((width, height), resample=Resampling.NEAREST)

    if image.width / image.height > 200:
        width, height = image.height * 180, image.height
        image = image.resize((width, height), resample=Resampling.NEAREST)

    if image.height / image.width > 200:
        width, height = image.width, image.width * 180
        image = image.resize((width, height), resample=Resampling.NEAREST)

    return image


def regularize(images, **kwargs) -> List["ImageObject"]:
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

        results.append(preprocess(image, **kwargs))

    return results


def _get_mm_inputs(images, processor):
    image_processor = getattr(processor, "image_processor")
    video_processor = getattr(processor, "video_processor", image_processor)
    input_dict = {"images": None}
    if len(images) != 0:
        images = regularize(
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


def translate_messages(messages, mm_path):
    res_messages = []
    images = []
    for i, message in enumerate(messages):
        role = message["role"]

        txt = ""
        for content in message["content"]:
            if content["type"] == "image":
                images.append(os.path.join(mm_path, content["image"]))
                txt += "<image>"
            elif content["type"] == "text":
                txt += content["text"]
            else:
                raise NotImplementedError("not supported type: {}".format(content["type"]))

        res_messages.append({"role": role, "content": txt})
    return res_messages, images


def process_messages(messages, images, processor, image_pad_token="<|image_pad|>"):
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
                    image_pad_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                ),
                1,
            )
            num_image_tokens += 1

        message["content"] = content

    if len(images) != num_image_tokens:
        raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))

    return messages, mm_inputs
