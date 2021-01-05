from dataset import PythonReposCachedDataset, preencode
from pathlib import Path
from tqdm import tqdm
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    T5Config, T5ForConditionalGeneration, T5TokenizerFast,
    AdamW,
)
from ccompletion.utils import (
    generate_samples, to_truncated_list, is_variant_same
)

import fire
import glob
import random
import shutil
import torch
import warnings

# Available GPT-2 and T5 variants
# GPT2: gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2
# T5: t5-small, t5-base, t5-large, t5-3b, t5-11b


def finetune(
    t5: bool = False,
    gpt2: bool = False,
    t5variant: str = 't5-base',
    gpt2variant: str = 'gpt2-medium',
    dataset_dir: str = 'repositories',
    batch_size: int = 8,
    block_size: int = 512,
    fp16: bool = True,
    steps_per_checkpoint: int = 100,
    start: int = 0,
    max_steps: int = 1e+15,
    learning_rate: float = 1e-5,
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
        'step_start': start,
        'max_steps': max_steps,
        'learning_rate': learning_rate,
    }

    if gpt2:
        print(f'Finetuning GPT-2 (variant={gpt2variant})...')
        finetune_gpt2(variant=gpt2variant, **kwargs)

    if t5:
        print(f'Finetuning T5 (variant={t5variant})...')
        finetune_t5(variant=t5variant, **kwargs)


def finetune_t5(
    variant: str = 't5-base',
    learning_rate: float = 1e-4,
    device: torch.device = 'cpu',
    dataset_dir: str = 'repositories',
    batch_size: int = 16,
    block_size: int = 2 << 8,
    use_fp16: bool = True,
    step_start: int = 0,
    steps_per_checkpoint: int = 100,
    max_steps: int = 1e+15,
    ckpt_path: str = 't5-finetuned',
):
    # preencode dataset for this model, batch size, and block size
    cache_file = Path(f'_t5_{batch_size}_{block_size}.pkl')
    if not cache_file.is_file():
        print(f'Cache file "{cache_file.name}" does not exist.')
        preencode('t5', dataset_dir=dataset_dir,
                  batch_size=batch_size, block_size=block_size)

    # instantiate model from the checkpoint (if exists), else variant
    if Path(ckpt_path).is_dir():
        print(f'Will load from checkpoint "{ckpt_path}".')
        model = T5ForConditionalGeneration.from_pretrained(ckpt_path)
        config = T5Config.from_pretrained(variant)
        assert is_variant_same(model.config, config)
    else:
        model = T5ForConditionalGeneration.from_pretrained(variant)

    # instantiate tokenizer based on the passed variant
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
    dataset = PythonReposCachedDataset(cache_file)

    # initialize model's optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # initialize tqdm progress bar (with optional offset)
    pbar = tqdm(total=len(dataset))
    pbar.update(step_start)

    # The goal of this finetuning is to let the model see each of the python source
    # files exactly once (and not by epochs)
    for i in range(step_start, min(len(dataset), max_steps)):
        # encode batch into their token IDS
        # split tensors since the model has a max length limit
        input_ids = tokenizer(
            dataset[i],
            return_tensors='pt',
            padding=True,
        ).input_ids.split(block_size, dim=1)

        for j, _input_ids in enumerate(input_ids):
            # convert tensor into a python list to generate labels for t5 finetuning
            _input_ids, _input_ids_mask, _labels = generate_samples(
                to_truncated_list(_input_ids)
            )

            # move input tensor to GPU
            _input_ids = torch.as_tensor(_input_ids, device=device)
            _input_ids_mask = torch.as_tensor(_input_ids_mask, device=device)
            _labels = torch.as_tensor(_labels, device=device)

            # compute loss
            loss = model(
                input_ids=_input_ids,
                attention_mask=_input_ids_mask,
                labels=_labels,
            ).loss

            # if loss turns out to be nan, then there's something wrong
            # with the inputs that was fed into the model so training should not continue
            if torch.isnan(loss):
                model.save_pretrained(f'{ckpt_path}-stopped-by-nan')
                pbar.write('Input batch that lead to nan loss:')
                pbar.write(str(dataset[i]))
                return

            # Write loss and continue training
            pbar.set_description(f'Step {i+1}-{j+1}: Loss={loss}')

            # delete input tensors to free memory in the GPU
            del _input_ids, _input_ids_mask, _labels

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save weights every `steps_per_checkpoint`
        if (i + 1) % steps_per_checkpoint == 0:
            [shutil.rmtree(path) for path in glob.glob(f'{ckpt_path}-*')]
            model.save_pretrained(f'{ckpt_path}-{i}')

        # mark step as done
        pbar.update(1)

    # save finetuned weights (final)
    model.save_pretrained(ckpt_path)
    print('Finished finetuning T5')


def finetune_gpt2(
    variant: str = 'gpt2',
    learning_rate: float = 1e-5,
    device: torch.device = 'cpu',
    dataset_dir: str = 'repositories',
    batch_size: int = 16,
    block_size: int = 2 << 8,
    use_fp16: bool = True,
    step_start: int = 0,
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

    # instantiate model from the checkpoint (if exists), else variant
    if Path(ckpt_path).is_dir():
        print(f'Will load from checkpoint "{ckpt_path}".')
        model = GPT2LMHeadModel.from_pretrained(ckpt_path)
        config = GPT2Config.from_pretrained(variant)
        assert is_variant_same(model.config, config)
    else:
        model = GPT2LMHeadModel.from_pretrained(variant)

    # instantiate tokenizer based on the passed variant
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
    dataset = PythonReposCachedDataset(cache_file)

    # initialize model's optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # initialize tqdm progress bar (with optional offset)
    pbar = tqdm(total=len(dataset))
    pbar.update(step_start)

    # The goal of this finetuning is to let the model see each of the python source
    # files exactly once (and not by epochs)
    for i in range(step_start, min(len(dataset), max_steps)):
        # encode batch into their token IDS
        # split tensors since the model has a max length limit
        encoding = tokenizer(dataset[i], return_tensors='pt', padding=True)
        input_ids = encoding['input_ids'].split(block_size, dim=1)
        attn_mask = encoding['attention_mask'].split(block_size, dim=1)

        for j in range(len(input_ids)):
            _input_ids = input_ids[j]
            _attn_mask = attn_mask[j]
            _labels = _input_ids.detach().clone()
            _labels[_labels == tokenizer.pad_token_id] = -100

            # skip batches with a width of less than 2
            # since we shift the positions of the tokens for their labels
            if _input_ids.shape[1] < 2:
                continue

            # move input tensors to GPU
            _input_ids = _input_ids.to(device)
            _attn_mask = _attn_mask.to(device)
            _labels = _labels.to(device)

            # compute loss
            loss = model(
                _input_ids,
                attention_mask=_attn_mask,
                labels=_labels
            ).loss

            # if loss turns out to be nan, then there's something wrong
            # with the inputs that was fed into the model so training should not continue
            if torch.isnan(loss):
                model.save_pretrained(f'{ckpt_path}-stopped-by-nan')
                pbar.write('Input batch that lead to nan loss:')
                pbar.write(str(dataset[i]))
                return

            # Write loss and continue training
            pbar.set_description(f'Step {i+1}-{j+1}: Loss={loss}')

            # delete input tensors to free memory in the GPU
            del _input_ids, _attn_mask, _labels

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # mark step as done
        pbar.update(1)

        # save weights every `steps_per_checkpoint`
        # delete old checkpoints to not overflow disk usage
        if (i + 1) % steps_per_checkpoint == 0:
            [shutil.rmtree(path) for path in glob.glob(f'{ckpt_path}-*')]
            model.save_pretrained(f'{ckpt_path}-{i}')

    # save finetuned weights (final)
    model.save_pretrained(ckpt_path)
    print('Finished finetuning GPT-2')


if __name__ == '__main__':
    # set random seed for reproducibility
    torch.manual_seed(7)
    random.seed(7)

    # check for CUDA availability
    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available. Training will be slow.")

    fire.Fire(finetune)
