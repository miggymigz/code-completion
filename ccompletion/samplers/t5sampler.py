from transformers import T5TokenizerFast, T5ForConditionalGeneration


def sampleT5(
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
