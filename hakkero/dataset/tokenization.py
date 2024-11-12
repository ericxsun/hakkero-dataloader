#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import torch
from hakkero.dataset.errors import TokenizationError
from hakkero.dataset.utils import IGNORE_INDEX, process_messages
from hakkero.dataset.utils import message_trans

def check_legacy(data):
    if not isinstance(data, dict):
        return False

    if any(s not in data for s in ("title", "summary", "abstract", "text", "question", "answer", "code")):
        return False

    return True


def check_message(data):
    if not isinstance(data, list):
        return False

    if not all(isinstance(d, dict) for d in data):
        return False

    if data[-1]["role"] != "assistant":
        return False

    if data[-2]["role"] != "user":
        return False

    return True


def check_preference(data):
    if not isinstance(data, dict):
        return False

    if "context" not in data or "chosen" not in data or "rejected" not in data:
        return False

    if not isinstance(data["chosen"], str) or not isinstance(data["rejected"], str):
        return False

    return check_message(data["context"])


def legacy(data, tokenizer, **kwargs):
    assert isinstance(data, dict), "wrong data format, expect {key: value}, " + f"but got {data}"

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
def huggingface_message(messages, tokenizer, processor, **kwargs):
    if not isinstance(messages, list) or not isinstance(messages[0], dict):
        raise ValueError("messages should be [{'role': 'xxx', 'content': 'xxx'}, ...]," + f" but got {messages}")

    assert hasattr(tokenizer, "apply_chat_template"), "tokenizer should have apply_chat_template"
    assert messages[-1]["role"] == "assistant", "messages[-1]['role'] should be 'assistant'"
    assert messages[-2]["role"] == "user", "messages[-2]['role'] should be 'user'"

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
def huggingface_preference(data, tokenizer, processor, **kwargs):
    assert hasattr(tokenizer, "apply_chat_template")
    assert data["context"][-1]["role"] == "user", "data['context'][-1]['role'] should be 'user'"

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
def role_message(messages, tokenizer, template):
    if not isinstance(messages, list) or not isinstance(messages[0], dict):
        raise ValueError("messages should be [{'role': 'xxx', 'content': 'xxx'}, ...]," + f" but got {messages}")

    assert messages[-1]["role"] == "assistant", "messages[-1]['role'] should be 'assistant'"
    assert messages[-2]["role"] == "user", "messages[-2]['role'] should be 'user'"

    assistant_start_ids = tokenizer.encode(
        template["assistant_start"], add_special_tokens=False, max_length=int(1e12), truncation=True
    )

    input, label, context = [], [], None
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


def chatml_message(messages, tokenizer, processor):
    return role_message(messages, tokenizer, chatml_role)


# data = {
#   "context": [{"role": "user", "content": xxx}, {"role": "assistant", "content": xxx}, ...],
#   "chosen": "xx",
#   "rejected": "xx"
# }
def role_preference(data, tokenizer, template):
    assert data["context"][-1]["role"] != "assistant"
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


def chatml_preference(data, tokenizer, processor):
    return role_preference(data, tokenizer, chatml_role)


# ----------------------------------------------------------------------------------------------------------------------
# --------MMLLM----------

qwen2_system = "You are a helpful assistant."

def qwen2_vl_message(messages, tokenizer, processor, file_path, **kwargs):
    template = chatml_role
    messages, images = message_trans(messages, file_path)
    if len(images) > 0:
        messages, mm_inputs = process_messages(messages,images,processor)
    if not isinstance(messages, list) or not isinstance(messages[0], dict):
        raise ValueError("messages should be [{'role': 'xxx', 'content': 'xxx'}, ...]," + f" but got {messages}")

    assert messages[-1]["role"] == "assistant", "messages[-1]['role'] should be 'assistant'"
    assert messages[-2]["role"] == "user", "messages[-2]['role'] should be 'user'"

    assistant_start_ids = tokenizer.encode(
        template["assistant_start"], add_special_tokens=False, max_length=int(1e12),truncation=True
    )
    input, label, context = [], [], template["system"].format(qwen2_system)
    # all_tempalte = ''
    for i, message in enumerate(messages, start=1):
        if message["role"] in ["user"]:
            text = template[message["role"]].format(message["content"])
            context = text if context is None else template["join"].join([context, text])
        elif message["role"] == "assistant":
            # only tokenize and append context right before assistant message
            # context after assistant message is not useful
            context = template["join"].join([context, template["assistant_start"]])
            #ids = tokenizer.encode(context, add_special_tokens=False, max_length=int(1e12), trucation=True)
            # all_tempalte += context
            ids = tokenizer.encode(context, add_special_tokens=False, max_length=int(1e12),truncation=True)
            input.extend(ids)

            label.extend([IGNORE_INDEX] * len(ids))

            ids = tokenizer.encode(
                template["assistant_start"] + message["content"] + template["assistant_end"],
                add_special_tokens=False,
                max_length=int(1e12),truncation=True
            )
            # all_tempalte += (message["content"] + template["assistant_end"])
            # a hack to avoid prepending space in the assistant response
            assert ids[: len(assistant_start_ids)] == assistant_start_ids
            input.extend(ids[len(assistant_start_ids) :])
            label.extend(ids[len(assistant_start_ids) :])
            context = ""
        else:
            raise ValueError(f"not supported role: {message['role']}")
    
    # return dict(input=torch.tensor(input[:-1]), label=torch.tensor(label[1:]))
    feature = dict(input=torch.tensor(input[:-1]), label=torch.tensor(label[1:]))
    feature.update(mm_inputs)
    return feature
