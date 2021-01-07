from transformers import T5TokenizerFast, T5ForConditionalGeneration

import numpy as np


def sampleT5v1(
    *,
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    sequence: str,
    beam_width: int = 5,
):
    # blanks should be written by 4 underscores
    if '____' not in sequence:
        raise ValueError('src should contain a blank (e.g. ____)')

    # replace blank by a sentinel token
    src = sequence.replace('____', '<extra_id_0>')

    # initialize model inputs and get outputs
    input_ids = tokenizer(src, add_special_tokens=True,
                          return_tensors='pt').input_ids
    outputs = model.generate(input_ids, num_beams=beam_width,
                             num_return_sequences=beam_width, max_length=10)

    # decode outputs
    blank_index = src.index('<extra_id_0>')
    src_prefix, src_suffix = src[:blank_index], src[blank_index+12:]

    def filter(output, end_token='<extra_id_1>'):
        # the first token is <pad> (index=0) and
        # the second token is <extra_id_0> (index=32099)
        _txt = tokenizer.decode(output[2:])
        _txt = _txt if end_token not in _txt else _txt[:_txt.index(end_token)]
        space = '' if src_prefix.endswith(' ') else ' '

        return src_prefix + space + _txt + src_suffix

    return list(map(filter, outputs))


def sampleT5v2(
    *,
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    sequence: str,
    beam_width: int = 5,
):
    # sample `beam_width` sequences
    input_ids = tokenizer(sequence, return_tensors='pt').input_ids
    outputs = model.generate(
        input_ids,
        num_beams=beam_width,
        num_return_sequences=beam_width,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # convert sequences' log probabilities
    probs = np.exp(outputs.sequences_scores).tolist()

    # each generated sequence starts with a pad token and ends with
    # an eos token which needs to be removed before decoding
    sequences = outputs.sequences[:, 1:-1]

    # decode generated sequences
    sequences = tokenizer.batch_decode(sequences)

    # each generated sequence may vary in lengths
    # thus the above removal of pad and eos token is not enough
    for i, sequence in enumerate(sequences):
        # discard eos token (plus more pad tokens if exists)
        rindex = sequence.rfind(tokenizer.eos_token)
        if rindex != -1:
            sequences[i] = sequence[:rindex]

        # single newline token prediction yields to an empty result
        if not sequences[i]:
            sequences[i] = '[newline]'

    return sorted(
        zip(probs, sequences),
        key=lambda pair: pair[0],
        reverse=True,
    )
