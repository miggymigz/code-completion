from pathlib import Path
from typing import List

import chardet
import numpy as np
import random


def read_source_file(path: Path) -> str:
    try:
        # attempt to read file using utf-8 encoding
        with open(path, 'r', encoding='utf-8', errors='strict') as fd:
            return fd.read().strip(), 'utf-8'
    except UnicodeDecodeError:
        # determine encoding of file using chardet module
        with open(path, 'rb') as fd:
            result = chardet.detect(fd.read())
            encoding = result['encoding']

            # rethrow error to caller if encoding could not be determined
            if encoding is None:
                raise

        # open file using the guessed encoding
        # errors raised on this one will be rethrown back to the caller
        with open(path, 'r', encoding=encoding, errors='strict') as fd:
            return fd.read().strip(), encoding


def is_variant_same(config1, config2):
    return (
        config1.vocab_size == config2.vocab_size
        and config1.n_ctx == config2.n_ctx
        and config1.n_embd == config2.n_embd
        and config1.n_head == config2.n_head
        and config1.n_inner == config2.n_inner
        and config1.n_layer == config2.n_layer
        and config1.n_positions == config2.n_positions
        and config1.model_type == config2.model_type
    )


def to_truncated_list(tensor):
    result = tensor.tolist()
    for i in range(len(result)):
        try:
            end = result[i].index(0)
            result[i] = result[i][:end]
        except ValueError:
            pass

    return result


def group(indices):
    result = []
    start = None
    last = None

    for idx in indices:
        if start is None:
            start = last = idx
        elif idx != last + 1:
            result.append((start, last + 1))
            start = last = idx
        else:
            last = idx

    if start is not None:
        result.append((start, last + 1))

    return result


def generate_samples(
    input_ids: List[List[int]],
    dprob: float = 0.15,
    sentinel_first_id: int = 32099,
    eos_token_id: int = 1,
    pad: bool = True,
    pad_token_id: int = 0,
):
    r"""Generates all input tensors needed for denoising pretraining objective for T5 models.

    Args:
      input_ids: the token IDs generated by a T5Tokenizer instance.
      dbprob: the percentage of tokens to be masked.
      sentinel_first_id: the token ID of the first sentinel token of a T5Tokenizer (e.g. <extra_id_0>)
      eos_token_id: the token ID of the end-of-sentence token of a T5Tokenizer (e.g. <eos>)
      pad: whether to pad the returned numpy array using `pad_token_id`
      pad_token_id: the token ID used to pad the remaining unused positions

    Return:
      input_ids: The token IDs used as the input of the model
      attn_mask_inputs: The attention mask of the `input_ids`
      labels: The token IDs used as the output of the model
      attn_mask_labels: The attention mask of the `labels`
    """

    # count the number of tokens for each of the sequences to drop
    n_word_dropouts = [
        int(len(sequence_ids) * dprob)
        for sequence_ids in input_ids
    ]

    # in case where there are very short sentences (so there are no tokens to drop),
    # we simply remove them from the batch
    to_remove = reversed([
        i for i, count in enumerate(n_word_dropouts)
        if count == 0
    ])
    for idx in to_remove:
        del n_word_dropouts[idx], input_ids[idx]

    # generate the random indices to drop in the input sequences (excluding special tokens)
    # group indices based on if they are sequential or not
    d_indices = [
        group(sorted(random.sample(range(len(s) - 1), n_word_dropouts[i])))
        for i, s in enumerate(input_ids)
    ]

    # accumulate input_ids and labels of the sequences in the batch
    out_input_ids = []
    out_labels = []
    for i, s_indices in enumerate(d_indices):
        # the input_ids and  labels of the current sequence
        s_input_ids = []
        s_labels = []
        offset = 0

        for j, (start, end) in enumerate(s_indices):
            # add for each sentinel token, it must be followed by the tokens it represents
            c_sentinel = sentinel_first_id - j
            s_input_ids.extend([*input_ids[i][offset:start], c_sentinel])
            s_labels.extend([c_sentinel, *input_ids[i][start:end]])

            # adjust offset for the input IDs
            offset = end

        # add the labels of the current sequence to the batch labels
        s_input_ids.extend([*input_ids[i][offset:]])
        s_labels.append(eos_token_id)
        out_input_ids.append(s_input_ids)
        out_labels.append(s_labels)

    # raw input_ids and labels (without padding, and not a tensor)
    outputs = out_input_ids, out_labels

    # we return a pytorch tensor with padding
    # (since tensor should have a consistent size)
    if pad:
        # determine the longest sequence length
        max_len_inputs = max(len(s) for s in out_input_ids)
        max_len_labels = max(len(s) for s in out_labels)

        # create attention masks tensors for both input_ids and labels
        attn_mask_inputs = np.ones((len(out_input_ids), max_len_inputs))

        # fill holes with pad tokens
        for i, s in enumerate(out_input_ids):
            if len(s) < max_len_inputs:
                delta = max_len_inputs - len(s)
                out_input_ids[i] += [pad_token_id] * delta
                attn_mask_inputs[i, -delta:] = 0

        # do the same with labels
        for i, s in enumerate(out_labels):
            if len(s) < max_len_labels:
                delta = max_len_labels - len(s)
                out_labels[i] += [-100] * delta

        # convert both lists into a pytorch tensor
        out_input_ids = np.array(out_input_ids)
        labels = np.array(out_labels)
        outputs = out_input_ids, attn_mask_inputs, labels

    return outputs
