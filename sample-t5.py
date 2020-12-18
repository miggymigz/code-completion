from pathlib import Path
from transformers import T5TokenizerFast, T5ForConditionalGeneration

import fire

# initialize model and tokenizer
t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5TokenizerFast.from_pretrained('t5-base')


def sample(inputfile: str = 'input.txt'):
    # verify input file's existence
    inputpath = Path(inputfile).expanduser()
    if not inputpath.is_file():
        raise FileNotFoundError('Input file not found')

    # read input file source code
    with open(inputpath, 'r', encoding='utf-8') as fd:
        src = fd.read()

    # blanks should be written by 4 underscores
    if '____' not in src:
        raise ValueError('src should contain a blank (e.g. ____)')

    # replace blank by a sentinel token
    src = src.replace('____', '<extra_id_0>')

    # initialize model inputs and get outputs
    input_ids = tokenizer(src, add_special_tokens=True,
                          return_tensors='pt').input_ids
    outputs = t5.generate(input_ids, num_beams=100,
                          num_return_sequences=10, max_length=10)

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

    print(list(map(filter, outputs)))


if __name__ == '__main__':
    fire.Fire(sample)
