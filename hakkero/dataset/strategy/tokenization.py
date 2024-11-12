#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import torch

from hakkero.dataset.image import process_messages
from hakkero.dataset.image import translate_messages
from hakkero.dataset.strategy.errors import TokenizationError
from hakkero.dataset.utils import IGNORE_INDEX


def legacy(data, tokenizer, **kwargs):
    context = "\n\n".join(
        [
            data[s].strip()
            for s in ("title", "summary", "abstract", "text", "question", "answer", "code")
            if s in data and data[s].strip()
        ]
    )

    target = data.get("label", "").strip()

    input = []
    label = []

    ids = tokenizer.encode(context, max_length=int(1e12), truncation=True)
    input.extend(ids)

    if target:
        if ids[-1] == tokenizer.eos_token_id:
            ids.pop()
        label.extend([IGNORE_INDEX for _ in ids])

        ids = tokenizer.encode(target, max_length=int(1e12), truncation=True)
        if ids[0] == tokenizer.bos_token_id:
            ids.pop(0)
        input.extend(ids)
        label.extend(ids)
    else:
        label.extend(ids)

    if len(input) <= 1:
        raise TokenizationError(
            "No valid keys in input, expect of: ('title', 'summary', 'abstract', 'text', 'question', 'answer', 'code')"
        )

    if kwargs.get("add_bos_token", False):
        if input[0] != tokenizer.bos_token_id:
            input = [tokenizer.bos_token_id] + input
        if label[0] != tokenizer.bos_token_id:
            label = [tokenizer.bos_token_id] + label

    if kwargs.get("add_eos_token", False):
        if input[-1] != tokenizer.eos_token_id:
            input = input + [tokenizer.eos_token_id]
        if label[-1] != tokenizer.eos_token_id:
            label = label + [tokenizer.eos_token_id]

    return dict(input=torch.tensor(input[:-1], dtype=torch.long), label=torch.tensor(label[1:], dtype=torch.long))


# ----------------------------------------------------------------------------------------------------------------------
# messages = [{"role": "user", "content": xxx}, {"role": "assistant", "content": xxx}, ...]
def huggingface_message(messages, tokenizer, **kwargs):
    assert hasattr(tokenizer, "apply_chat_template"), "tokenizer should have apply_chat_template"

    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], add_generation_prompt=True
    ) != tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], add_generation_prompt=False
    ), "add_generation_prompt does not take effect, please modify tokenizer.chat_template"

    context_ids = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)

    # hack: separate encoding of the context and response will always lead to prefix space in the response
    response_ids_with_prefix = tokenizer.apply_chat_template(messages[-2:], add_generation_prompt=False)
    prefix_ids = tokenizer.apply_chat_template(messages[-2:-1], add_generation_prompt=True)
    response_ids = response_ids_with_prefix[len(prefix_ids) :]

    input = context_ids + response_ids
    label = [IGNORE_INDEX for _ in context_ids] + response_ids

    return dict(input=torch.tensor(input[:-1], dtype=torch.long), label=torch.tensor(label[1:], dtype=torch.long))


# data = {
#   "context": [{"role": "user", "content": xxx}, {"role": "assistant", "content": xxx}, ...],
#   "chosen": "xx",
#   "rejected": "xx"
# }
def huggingface_preference(data, tokenizer, **kwargs):
    assert hasattr(tokenizer, "apply_chat_template")

    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], add_generation_prompt=True
    ) != tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], add_generation_prompt=False
    ), "add_generation_prompt does not take effect, please modify tokenizer.chat_template"

    context_ids = tokenizer.apply_chat_template(data["context"], add_generation_prompt=True)

    # hack: separate encoding of the context and response will always lead to prefix space in the response
    prefix_ids = tokenizer.apply_chat_template(data["context"][-1:], add_generation_prompt=True)

    inputs = dict(chosen=[], rejected=[])
    labels = dict(chosen=[], rejected=[])

    for key in ("chosen", "rejected"):
        inputs[key].extend(context_ids)
        labels[key].extend(IGNORE_INDEX for _ in context_ids)
        response_ids_with_prefix = tokenizer.apply_chat_template(
            data["context"][-1:] + [{"role": "assistant", "content": data[key]}], add_generation_prompt=False
        )

        assert response_ids_with_prefix[: len(prefix_ids)] == prefix_ids

        response_ids = response_ids_with_prefix[len(prefix_ids) :]

        inputs[key].extend(response_ids)
        labels[key].extend(response_ids)

    return {
        "inputs": {key: torch.tensor(value[:-1]) for key, value in inputs.items()},
        "labels": {key: torch.tensor(value[1:]) for key, value in labels.items()},
    }


# ----------------------------------------------------------------------------------------------------------------------
chatml_role = {
    "join": "\n",
    "user": "<|im_start|>user\n{}<|im_end|>",
    "system": "<|im_start|>system\n{}<|im_end|>",
    "assistant": "<|im_start|>assistant\n{}<|im_end|>",
    "assistant_start": "<|im_start|>assistant\n",
    "assistant_end": "<|im_end|>",
}


# messages = [{"role": "user", "content": xxx}, {"role": "assistant", "content": xxx}, ...]
def role_message(messages, tokenizer, template, context=None):
    assistant_start_ids = tokenizer.encode(
        template["assistant_start"], add_special_tokens=False, max_length=int(1e12), truncation=True
    )

    input, label, context = [], [], context
    for i, message in enumerate(messages, start=1):
        if message["role"] in ["system", "user"]:
            text = template[message["role"]].format(message["content"])
            context = text if context is None else template["join"].join([context, text])
        elif message["role"] == "assistant":
            # only tokenize and append context right before assistant message
            # context after assistant message is not useful
            context = template["join"].join([context, template["assistant_start"]])
            ids = tokenizer.encode(context, add_special_tokens=False, max_length=int(1e12), truncation=True)
            input.extend(ids)

            label.extend([IGNORE_INDEX for _ in ids])

            ids = tokenizer.encode(
                template["assistant_start"] + message["content"] + template["assistant_end"],
                add_special_tokens=False,
                max_length=int(1e12),
                truncation=True,
            )

            # a hack to avoid prepending space in the assistant response
            assert ids[: len(assistant_start_ids)] == assistant_start_ids
            input.extend(ids[len(assistant_start_ids) :])
            label.extend(ids[len(assistant_start_ids) :])
            context = ""
        else:
            raise ValueError(f"not supported role: {message['role']}")

    return dict(input=torch.tensor(input[:-1]), label=torch.tensor(label[1:]))


def chatml_message(messages, tokenizer, **kwargs):
    return role_message(messages, tokenizer, chatml_role)


# data = {
#   "context": [{"role": "user", "content": xxx}, {"role": "assistant", "content": xxx}, ...],
#   "chosen": "xx",
#   "rejected": "xx"
# }
def role_preference(data, tokenizer, template):
    assistant_start_ids = tokenizer.encode(
        template["assistant_start"], add_special_tokens=False, max_length=int(1e12), truncation=True
    )
    inputs = dict(chosen=[], rejected=[])
    labels = dict(chosen=[], rejected=[])

    context = template["join"].join(
        [template[message["role"]].format(message["content"]) for message in data["context"]]
        + [template["assistant_start"]]
    )
    context_ids = tokenizer.encode(context, add_special_tokens=False, max_length=int(1e12), truncation=True)

    for key in ("chosen", "rejected"):
        inputs[key].extend(context_ids)
        labels[key].extend(IGNORE_INDEX for _ in context_ids)
        response_ids_with_prefix = tokenizer.encode(
            template["assistant_start"] + data[key] + template["assistant_end"],
            add_special_tokens=False,
            max_length=int(1e12),
            truncation=True,
        )
        assert response_ids_with_prefix[: len(assistant_start_ids)] == assistant_start_ids
        response_ids = response_ids_with_prefix[len(assistant_start_ids) :]
        inputs[key].extend(response_ids)
        labels[key].extend(response_ids)

    return {
        "inputs": {key: torch.tensor(value[:-1]) for key, value in inputs.items()},
        "labels": {key: torch.tensor(value[1:]) for key, value in labels.items()},
    }


def chatml_preference(data, tokenizer, **kwargs):
    return role_preference(data, tokenizer, chatml_role)


# ----------------------------------------------------------------------------------------------------------------------
# for MM LLM

qwen2_system = "You are a helpful assistant."


def chatml_qwen2_vl_message(messages, tokenizer, processor, path, **kwargs):
    messages, images = translate_messages(messages, path)
    mm_inputs = None
    if len(images) > 0:
        messages, mm_inputs = process_messages(messages, images, processor)

    msg = role_message(messages, tokenizer, chatml_role, context=chatml_role["system"].format(qwen2_system))
    if mm_inputs is not None:
        msg.update(mm_inputs)

    return msg
