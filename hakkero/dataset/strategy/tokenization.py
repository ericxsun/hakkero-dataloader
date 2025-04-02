#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import re

import torch

from hakkero.dataset.image import process_messages
from hakkero.dataset.image import translate_messages
from hakkero.dataset.strategy.errors import TokenizationError
from hakkero.dataset.strategy.keep_thinks import change_chat_template_with_keep_think_remap
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


def remove_ignore(content, ignore):
    if ignore is None:
        return content

    re_ignore = re.compile(f"(?:{ignore})$")
    return re_ignore.split(content)[0]


def _override_tokenizer(kwargs, tokenizer):
    old_chat_template = tokenizer.chat_template
    keep_think = kwargs.pop("keep_think", "no").lower()
    new_chat_template, keep_think = change_chat_template_with_keep_think_remap(old_chat_template, keep_think)
    tokenizer.chat_template = new_chat_template

    return keep_think, tokenizer, old_chat_template


# ----------------------------------------------------------------------------------------------------------------------
# keep_think: no - not keep <think>xx</think>, last - keep <think>xx</think> in last turn, all - keep <think>xx</think> in all turns
_tokenize_kwargs = {
    "padding": False,
    "truncation": False,
    "max_length": None,
    "add_special_tokens": False,
    "return_tensors": None,
}


def _process_single_turn(uid, is_last_turn, messages, tokenizer, keep_think, **kwargs):
    keep_think_in_context = keep_think == "all"
    keep_think_in_resp = keep_think == "all" or (keep_think == "last" and is_last_turn)

    aid = uid + 1

    text_context_ids = tokenizer.apply_chat_template(
        messages[:aid], add_generation_prompt=True, tokenize=False, keep_think=keep_think_in_context
    )
    text_context_ids = remove_ignore(text_context_ids, "<think>\n")
    context_ids = tokenizer(text_context_ids, **_tokenize_kwargs)["input_ids"]

    text_resp_ids_with_prefix = tokenizer.apply_chat_template(
        messages[: aid + 1], add_generation_prompt=False, tokenize=False, keep_think=keep_think_in_resp
    )

    text_prefix_ids = tokenizer.apply_chat_template(
        messages[:aid], add_generation_prompt=True, tokenize=False, keep_think=keep_think_in_resp
    )
    text_prefix_ids = remove_ignore(text_prefix_ids, "<think>\n")

    assert text_resp_ids_with_prefix[: len(text_prefix_ids)] == text_prefix_ids

    resp_ids = tokenizer(text_resp_ids_with_prefix[len(text_prefix_ids) :], **_tokenize_kwargs)["input_ids"]

    cur_input = context_ids + resp_ids
    cur_label = [IGNORE_INDEX for _ in context_ids] + resp_ids

    return cur_input, cur_label


# messages = [{"role": "user", "content": xxx}, {"role": "assistant", "content": xxx}, ...]
def huggingface_message(messages, tokenizer, **kwargs):
    keep_think, tokenizer, old_chat_template = _override_tokenizer(kwargs, tokenizer)

    n_turns = int((len(messages) if messages[0]["role"] == "user" else len(messages) - 1) / 2)

    input, label = [], []

    uid = next((i for i, v in enumerate(messages) if v["role"] == "user"), -1)
    cur_turn = 1

    while uid < len(messages):
        is_last_turn = cur_turn == n_turns

        cur_input, cur_label = _process_single_turn(uid, is_last_turn, messages, tokenizer, keep_think, **kwargs)

        input = input + cur_input[len(input) :]
        label = label + cur_label[len(label) :]

        uid += 2
        cur_turn += 1

    tokenizer.chat_template = old_chat_template

    return dict(input=torch.tensor(input[:-1], dtype=torch.long), label=torch.tensor(label[1:], dtype=torch.long))


# data = {
#   "context": [
#       {"role": "user", "content": xxx},
#       {"role": "assistant", "content": xxx},
#       ...
#       {"role": "user", "content": xxx}
#   ],
#   "chosen": "xx",
#   "rejected": "xx"
# }
def huggingface_preference(data, tokenizer, **kwargs):
    keep_think, tokenizer, old_chat_template = _override_tokenizer(kwargs, tokenizer)

    keep_think_in_context = keep_think == "all"
    keep_think_in_resp = keep_think == "all" or keep_think == "last"  # keep think in chosen/rejected

    text_context_ids = tokenizer.apply_chat_template(
        data["context"], add_generation_prompt=True, tokenize=False, keep_think=keep_think_in_context
    )
    text_context_ids = remove_ignore(text_context_ids, "<think>\n")
    context_ids = tokenizer(text_context_ids, **_tokenize_kwargs)["input_ids"]

    text_prefix_ids = tokenizer.apply_chat_template(
        data["context"][-1:], add_generation_prompt=True, tokenize=False, keep_think=keep_think_in_resp
    )
    text_prefix_ids = remove_ignore(text_prefix_ids, "<think>\n")

    inputs = dict(chosen=[], rejected=[])
    labels = dict(chosen=[], rejected=[])

    for key in ("chosen", "rejected"):
        inputs[key].extend(context_ids)
        labels[key].extend(IGNORE_INDEX for _ in context_ids)

        text_response_ids_with_prefix = tokenizer.apply_chat_template(
            data["context"][-1:] + [{"role": "assistant", "content": data[key]}],
            add_generation_prompt=False,
            tokenize=False,
            keep_think=keep_think_in_resp,
        )

        assert text_response_ids_with_prefix[: len(text_prefix_ids)] == text_prefix_ids

        response_ids = tokenizer(text_response_ids_with_prefix[len(text_prefix_ids) :], **_tokenize_kwargs)["input_ids"]

        inputs[key].extend(response_ids)
        labels[key].extend(response_ids)

    tokenizer.chat_template = old_chat_template

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
