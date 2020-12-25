from dataset import PythonReposCachedDataset, preencode
from pathlib import Path
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    T5ForConditionalGeneration, T5TokenizerFast,
    AdamW, Adafactor,
)

import fire
import random
import torch
import warnings

# Available GPT-2 and T5 variants
# GPT2: gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2
# T5: t5-small, t5-base, t5-large, t5-3b, t5-11b


def finetune(
    t5: bool = False,
    gpt2: bool = False,
    t5variant: str = 't5-large',
    gpt2variant: str = 'gpt2-large',
    dataset_dir: str = 'repositories',
    batch_size: int = 16,
    block_size: int = 2 << 8,
    fp16: bool = True,
    steps_per_checkpoint: int = 10,
    max_steps: int = 1e+15,
):
    # instantiate device to be used for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pack common keyword arguments
    kwargs = {
        'device': device,
        'dataset_dir': dataset_dir,
        'batch_size': batch_size,
        'block_size': block_size,
        'use_fp16': fp16,
        'steps_per_checkpoint': steps_per_checkpoint,
        'max_steps': max_steps,
    }

    if gpt2:
        print(f'Finetuning GPT-2 (variant={gpt2variant})...')
        finetune_gpt2(variant=gpt2variant, **kwargs)

    if t5:
        print(f'Finetuning T5 (variant={t5variant})...')
        finetune_t5(variant=t5variant, **kwargs)


def finetune_t5(
    variant: str = 't5-base',
    learning_rate: float = 0.001,
    device: torch.device = 'cpu',
    dataset_dir: str = 'repositories',
    batch_size: int = 16,
    block_size: int = 2 << 8,
    use_fp16: bool = True,
    steps_per_checkpoint: int = 10,
    max_steps: int = 1e+15,
    ckpt_path: str = 't5-finetuned',
):
    # preencode dataset for this model, batch size, and block size
    cache_file = Path(f'_t5_{batch_size}_{block_size}.pkl')
    if not cache_file.is_file():
        print(f'Cache file "{cache_file.name}" does not exist.')
        preencode('t5', dataset_dir=dataset_dir,
                  batch_size=batch_size, block_size=block_size)

    # instantiate pretrained tokenizer and model
    model = T5ForConditionalGeneration.from_pretrained(variant)
    tokenizer = T5TokenizerFast.from_pretrained(variant)

    # put model on cuda device and set it to training mode
    model.to(device)
    model.train()

    # use half-precision format during training
    # which leads to shorter training time and lower memory requirements
    # also enabling larger batch sizes
    if torch.cuda.is_available() and use_fp16:
        model.half()

    # retrieve python repositories dataset
    dataset = PythonReposCachedDataset(cache_file, max_steps=max_steps)

    # initialize model's optimizer
    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        relative_step=False
    )

    # The goal of this finetuning is to let the model see each of the python source
    # files exactly once (and not by epochs)
    pbar = tqdm(dataset)
    for i, batch in enumerate(pbar):
        # encode batch into their token IDS
        # split tensors since the model has a max length limit
        input_ids = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
        ).input_ids.split(block_size, dim=1)

        for j, _input_ids in enumerate(input_ids):
            # convert tensor into a python list to generate labels for t5 finetuning
            _input_ids, _input_ids_mask, _labels, _labels_mask = generate_samples(
                to_truncated_list(_input_ids)
            )

            # move input tensor to GPU
            _input_ids = _input_ids.to(device)
            _input_ids_mask = _input_ids_mask.to(device)
            _labels = _labels.to(device)
            _labels_mask = _labels_mask.to(device)

            # compute loss
            loss = model(
                input_ids=_input_ids,
                attention_mask=_input_ids_mask,
                labels=_labels,
                decoder_attention_mask=_labels_mask,
            ).loss

            # if loss turns out to be nan, then there's something wrong
            # with the inputs that was fed into the model so training should not continue
            if torch.isnan(loss):
                model.save_pretrained(f'{ckpt_path}-stopped-by-nan')
                pbar.write('Input batch that lead to nan loss:')
                pbar.write(str(batch))
                return

            # Write loss and continue training
            pbar.write(f'Step {i+1}-{j+1}: Loss={loss}')

            # delete input tensors to free memory in the GPU
            del _input_ids, _input_ids_mask, _labels, _labels_mask

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save weights every `steps_per_checkpoint`
        if (i + 1) % steps_per_checkpoint == 0:
            model.save_pretrained(f'{ckpt_path}-i')

    # save finetuned weights (final)
    model.save_pretrained(ckpt_path)
    print('Finished finetuning T5')


def finetune_gpt2(
    variant: str = 'gpt2',
    device: torch.device = 'cpu',
    dataset_dir: str = 'repositories',
    batch_size: int = 16,
    block_size: int = 2 << 8,
    use_fp16: bool = True,
    steps_per_checkpoint: int = 10,
    max_steps: int = 1e+15,
    ckpt_path: str = 'gpt2-finetuned',
):
    # preencode dataset for this model, batch size, and block size
    cache_file = Path(f'_gpt2_{batch_size}_{block_size}.pkl')
    if not cache_file.is_file():
        print(f'Cache file "{cache_file.name}" does not exist.')
        preencode('gpt2', dataset_dir=dataset_dir,
                  batch_size=batch_size, block_size=block_size)

    # instantiate pretrained tokenizer and model
    model = GPT2LMHeadModel.from_pretrained(variant)
    tokenizer = GPT2TokenizerFast.from_pretrained(variant)

    # put model on cuda device and set it to training mode
    model.to(device)
    model.train()

    # use half-precision format during training
    # which leads to shorter training time and lower memory requirements
    # also enabling larger batch sizes
    if torch.cuda.is_available() and use_fp16:
        model.half()

    # Padding tokens were not used during the pre-training of GPT and GPT-2, therefore they have none.
    # An attention mask should be specified so that the model won't attend to padded indices.
    # A padding token is set here anyway because we want the tokenizer to return tensors.
    tokenizer.pad_token = tokenizer.eos_token

    # retrieve python repositories dataset
    dataset = PythonReposCachedDataset(cache_file, max_steps)

    # initialize model's optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # The goal of this finetuning is to let the model see each of the python source
    # files exactly once (and not by epochs)
    pbar = tqdm(dataset)
    for i, batch in enumerate(pbar):
        # encode batch into their token IDS
        # split tensors since the model has a max length limit
        encoding = tokenizer(batch, return_tensors='pt', padding=True)
        input_ids = encoding['input_ids'].split(block_size, dim=1)
        attn_mask = encoding['attention_mask'].split(block_size, dim=1)

        for j in range(len(input_ids)):
            _input_ids = input_ids[j]
            _attn_mask = attn_mask[j]

            # skip batches with a width of less than 2
            # since we shift the positions of the tokens for their labels
            if _input_ids.shape[1] < 2:
                continue

            # move input tensors to GPU
            _input_ids = _input_ids.to(device)
            _attn_mask = _attn_mask.to(device)

            # compute loss
            loss = model(
                _input_ids,
                attention_mask=_attn_mask,
                labels=_input_ids
            ).loss

            # if loss turns out to be nan, then there's something wrong
            # with the inputs that was fed into the model so training should not continue
            if torch.isnan(loss):
                model.save_pretrained(f'{ckpt_path}-stopped-by-nan')
                pbar.write('Input batch that lead to nan loss:')
                pbar.write(str(batch))
                return

            # Write loss and continue training
            pbar.write(f'Step {i+1}-{j+1}: Loss={loss}')

            # delete input tensors to free memory in the GPU
            del _input_ids, _attn_mask

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save weights every `steps_per_checkpoint`
        if (i + 1) % steps_per_checkpoint == 0:
            model.save_pretrained(f'{ckpt_path}-i')

    # save finetuned weights (final)
    model.save_pretrained(ckpt_path)
    print('Finished finetuning GPT-2')


def generate_samples(input_ids, dprob=0.15, sentinel_first_id=32099, eos_token_id=1, pad=True, pad_token_id=0):
    # count the number of tokens for each of the sequences to drop
    n_word_dropouts = [
        int(len(sequence_ids) * dprob)
        for sequence_ids in input_ids
    ]

    # in case where there are very short sentences (so there are no tokens to drop),
    # we simply remove them from the batch
    to_remove = [i for i, count in enumerate(n_word_dropouts) if count == 0]
    for idx in to_remove:
        del n_word_dropouts[idx], input_ids[idx]

    # generate the random indices to drop in the input sequences (excluding special tokens)
    # group indices based on if they are sequential or not
    d_indices = [
        group(sorted(random.choices(range(len(s) - 1), k=n_word_dropouts[i])))
        for i, s in enumerate(input_ids)
    ]

    # accumulate labels of the sequences in the batch
    labels = []
    for i, s_indices in enumerate(d_indices):
        # the labels of the current sequence
        s_labels = []
        for j, (start, end) in enumerate(s_indices):
            # add for each sentinel token, it must be followed by the tokens it represents
            s_labels.extend([sentinel_first_id - j, *input_ids[i][start:end]])

            # we delete the tokens from the input_ids (since we already put it in labels)
            # and we replace those tokens with the sentinel token
            del input_ids[i][start:end]
            input_ids[i].insert(start, sentinel_first_id - j)

        # add the labels of the current sequence to the batch labels
        s_labels.append(eos_token_id)
        labels.append(s_labels)

    # raw input_ids and labels (without padding, and not a tensor)
    outputs = input_ids, labels

    # we return a pytorch tensor with padding
    # (since tensor should have a consistent size)
    if pad:
        # determine the longest sequence length
        max_len_inputs = max([len(s) for s in input_ids])
        max_len_labels = max([len(s) for s in labels])

        # create attention masks tensors for both input_ids and labels
        attn_mask_inputs = torch.ones(len(input_ids), max_len_inputs)
        attn_mask_labels = torch.ones(len(labels), max_len_labels)

        # fill holes with pad tokens
        for i, s in enumerate(input_ids):
            if len(s) < max_len_inputs:
                delta = max_len_inputs - len(s)
                input_ids[i] += [pad_token_id] * delta
                attn_mask_inputs[i, -delta:] = 0

        # do the same with labels
        for i, s in enumerate(labels):
            if len(s) < max_len_labels:
                delta = max_len_labels - len(s)
                labels[i] += [pad_token_id] * delta
                attn_mask_labels[i, -delta:] = 0

        # convert both lists into a pytorch tensor
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        outputs = input_ids, attn_mask_inputs, labels, attn_mask_labels

    return outputs


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


def to_truncated_list(tensor):
    result = tensor.tolist()
    for i in range(len(result)):
        try:
            end = result[i].index(0)
            result[i] = result[i][:end]
        except ValueError:
            pass

    return result


if __name__ == '__main__':
    # set random seed for reproducibility
    torch.manual_seed(7)
    random.seed(7)

    # check for CUDA availability
    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available. Training will be slow.")

    fire.Fire(finetune)
