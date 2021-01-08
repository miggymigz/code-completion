from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from typing import List

import torch
import numpy as np

ENDING_TOKENS = ('\n', '<|endoftext|>')


def strip_end_tokens(sequence):
    for token in ENDING_TOKENS:
        sequence = sequence.rstrip(token)

    return sequence


def beam_step(
    *,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    sequences: str,
    k: int,
    probs: List[int] = None,
):
    if probs is not None:
        assert len(sequences) == len(probs)

    with torch.no_grad():
        # for sequences of different lengths, we may need to add padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(sequences, return_tensors='pt', padding=True)
        logits = model(**inputs).logits
        log_probs = torch.nn.LogSoftmax(dim=-1)(logits[:, -1, :])

    # add probs tensor (if given)
    if probs is not None:
        assert len(probs) == log_probs.size(0)

        # accumulate log probabilities by adding the past and current probs
        for i, p in enumerate(probs):
            log_probs[i] += p

    # flatten log probabilities to get the top k probs
    _k = k if probs is None else len(probs)
    topk = torch.topk(log_probs.view(-1), _k * 2)
    top_indices = topk.indices.view(-1).tolist()
    top_log_probs = topk.values.view(-1).tolist()

    # concat top k tokens to original sequences
    unique_sequences = set()
    new_sequences = []
    new_probs = []

    for i, index in enumerate(top_indices):
        seq_index, token_index = divmod(index, logits.size(-1))
        new_sequence = sequences[seq_index] + tokenizer.decode([token_index])
        new_unique_sequence = strip_end_tokens(new_sequence)

        if new_unique_sequence not in unique_sequences:
            unique_sequences.add(new_unique_sequence)
            new_sequences.append(new_sequence)
            new_probs.append(top_log_probs[i])

        if len(unique_sequences) == _k:
            break

    return new_sequences, new_probs


def divide(sequences: List[str], probs: List[int], n_steps, max_steps):
    finished_sequences = []
    finished_probs = []
    unfinished_sequences = []
    unfinished_probs = []

    for i, s in enumerate(sequences):
        # return sequences as they are if iter limit is reached
        if n_steps == max_steps:
            finished_sequences.append(s)
            finished_probs.append(probs[i])
            continue

        if s.endswith(ENDING_TOKENS):
            finished_sequences.append(s)
            finished_probs.append(probs[i])
        else:
            unfinished_sequences.append(s)
            unfinished_probs.append(probs[i])

    return finished_sequences, finished_probs, unfinished_sequences, unfinished_probs


def sampleGPT2(
    *,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    sequence: str,
    beam_width: int = 5,
    max_steps: int = 10,
):
    # do one step of beam search to get `beam_width` sequences
    sequences, probs = beam_step(
        model=model,
        tokenizer=tokenizer,
        sequences=[sequence],
        k=beam_width
    )
    fs, fp, sequences, probs = divide(sequences, probs, 1, max_steps)

    # we stop generating if one of the sequences
    # already contain an ending token (e.g. '\n')
    # or we got to the maximum number of steps
    n_steps = 1
    while sequences:
        sequences, probs = beam_step(
            model=model,
            tokenizer=tokenizer,
            sequences=sequences,
            probs=probs,
            k=beam_width,
        )
        _fs, _fp, sequences, probs = divide(
            sequences, probs, n_steps, max_steps)

        # accumulate finished sequences and probs
        fs.extend(_fs)
        fp.extend(_fp)
        n_steps += 1

    # the accumulated probabilities are log probabilities
    # we convert them back into normal probabilities for easier interpretation
    fp = torch.exp(torch.tensor(fp)).tolist()

    # due to some processing steps in the decoding step
    # the probability of the sequences are not in order
    # so we sort them before returning it to the caller
    return sorted(zip(fp, fs), key=lambda pair: pair[0], reverse=True)


def sampleGPT2v2(
    *,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    sequence: str,
    beam_width: int = 5,
    eos_token_id: int = 198,
):
    # make sure eos_token_id is correctly set
    assert tokenizer.decode(eos_token_id) == '\n'

    # sample `beam_width` sequences
    input_ids = tokenizer(sequence, return_tensors='pt').input_ids
    aux_beam_width = beam_width + int(beam_width * 0.75)
    outputs = model.generate(
        input_ids,
        do_sample=False,
        num_beams=aux_beam_width,
        num_return_sequences=aux_beam_width,
        early_stopping=True,
        eos_token_id=eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
        max_length=input_ids.size(-1) + 20,
    )

    # convert sequences' log probabilities
    probs = np.exp(outputs.sequences_scores).tolist()

    # decode generated sequences
    sequences = tokenizer.batch_decode(
        outputs.sequences, skip_special_tokens=True)

    # only preserve the last lines
    for i in range(len(sequences)):
        sequences[i] = sequences[i].strip()

        try:
            ridx = sequences[i].rindex('\n')
        except ValueError:
            pass
        else:
            print('BEFORE: ', sequences[i])
            sequences[i] = sequences[i][ridx+1:]
            print('AFTER: ', sequences[i])

    # remove duplicate suggestions/recommendations
    for i in reversed(range(len(sequences))):
        # retrieve current sequence
        sequence = sequences[i]

        # remove duplicate suggestions
        try:
            idx = sequences.index(sequence)
        except ValueError:
            pass
        else:
            if i != idx:
                del probs[i], sequences[i]
            continue

        # single newline token prediction yields to an empty result
        if not sequence:
            sequences[i] = '[newline]'

    return sorted(
        zip(probs, sequences),
        key=lambda pair: pair[0],
        reverse=True,
    )
