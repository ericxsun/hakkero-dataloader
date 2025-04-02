#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from hakkero.dataset.dataloader import PadLoader
from hakkero.dataset.dataloader import PreferencePadLoader
from hakkero.dataset.dataloader import PreferenceUnpadLoader
from hakkero.dataset.dataloader import UnpadLoader
from hakkero.dataset.image import process_messages
from hakkero.dataset.mixed_dataset import get_dataset
from hakkero.dataset.utils import IGNORE_INDEX

_unpad_loader = {False: UnpadLoader, True: PreferenceUnpadLoader}
_pad_loader = {False: PadLoader, True: PreferencePadLoader}


def get_data(
    config,
    dp_rank,
    dp_world_size,
    tokenizer,
    batch_size,
    max_length,
    num_epochs=10,
    homogeneous=False,
    seed=9527,
    n_workers=2,
    use_unpad_data=False,
    use_unpad_in_pad=False,
    use_cu_seqlens=False,
    st_segment=None,
    st_tokenize=None,
    pad_with_ignore_index=False,
    add_bos_token=True,
    add_eos_token=True,
    norm_weight_with_n_targets=False,
    keep_think="no",
    processor=None,
    **kwargs
):
    dataset = get_dataset(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        num_epochs=num_epochs,
        max_length=max_length,
        homogeneous=homogeneous,
        seed=seed,
        rank=dp_rank,
        world_size=dp_world_size,
        n_workers=n_workers,
        # segment & tokenize strategy
        st_segment=st_segment,
        st_tokenize=st_tokenize,
        #
        add_bos_token=add_bos_token,
        add_eos_token=add_eos_token,
        #
        norm_weight_with_n_targets=norm_weight_with_n_targets,
        keep_think=keep_think,
        **kwargs
    )

    is_preference = "preference" in list(config.values())[0].get("strategy", {}).get("st_tokenize", "")
    is_preference = is_preference or (st_tokenize is not None and "preference" in st_tokenize)

    if use_unpad_data:
        loader = _unpad_loader[is_preference](dataset, max_total_length=batch_size * max_length)
    else:
        if pad_with_ignore_index:
            padding_id = IGNORE_INDEX
        elif tokenizer.pad_token_id is None:
            padding_id = tokenizer.eos_token_id
        else:
            padding_id = tokenizer.pad_token_id

        loader = _pad_loader[is_preference](
            dataset, batch_size=batch_size, padding_id=padding_id, unpad=use_unpad_in_pad
        )

    if use_unpad_data or use_unpad_in_pad:
        if use_cu_seqlens:
            forward_keys = ["input_ids", "cu_seqlens"]
        else:
            forward_keys = ["input_ids", "position_ids"]
    else:
        forward_keys = ["input_ids", "attention_mask"]

    return dataset, loader, forward_keys
