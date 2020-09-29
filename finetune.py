from dataset import PythonReposDataset
from pathlib import Path
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    T5ForConditionalGeneration, T5Tokenizer,
    AdamW,
)

import fire
import torch
import warnings


def finetune(
    checkpoint_dir: str = 'checkpoints',
    dataset_dir: str = 'repositories',
    batch_size: int = 16,
    fp16: bool = True,
):
    # instantiate device to be used for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pack common keyword arguments
    kwargs = {
        'device': device,
        'checkpoint_dir': checkpoint_dir,
        'dataset_dir': dataset_dir,
        'batch_size': batch_size,
        'use_fp16': fp16,
    }

    # finetune GPT-2 and T5
    finetune_gpt2(variant='gpt2', **kwargs)
    finetune_t5(variant='t5', **kwargs)


def finetune_t5(
    variant: str = 't5-base',
    device: torch.device = 'cpu',
    checkpoint_dir: str = 'checkpoints',
    dataset_dir: str = 'repositories',
    batch_size: int = 16,
    use_fp16: bool = True,
):
    # instantiate pretrained tokenizer and model
    model = T5ForConditionalGeneration.from_pretrained(variant)
    tokenizer = T5Tokenizer.from_pretrained(variant)

    # put model on cuda device and set it to training mode
    model.to(device)
    model.train()

    # use half-precision format during training
    # which leads to shorter training time and lower memory requirements
    # also enabling larger batch sizes
    if use_fp16:
        model.half()

    # retrieve python repositories dataset
    block_size = 2 << 7
    dataset = PythonReposDataset(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        count_fn=lambda x: len(tokenizer(x)['input_ids']),
        block_size=block_size,
    )

    # initialize model's optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # The goal of this finetuning is to let the model see each of the python source
    # files exactly once (and not by epochs)
    for i, batch in enumerate(dataset):
        # encode batch into their token IDS
        input_ids = tokenizer.encode(batch, return_tensors='pt', padding=True)

        # split tensors since the model has a max length limit
        input_ids = input_ids.transpose(0, 1).split(block_size)

        for j in range(len(input_ids)):
            _input_ids = input_ids[j].transpose(0, 1).to(device)

            # compute loss
            loss, _, _ = model(input_ids=_input_ids, labels=_input_ids)
            print(f'Step {i+1}-{j+1}: Loss={loss}')

            # delete input tensors to free memory in the GPU
            del _input_ids

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save finetuned weights
    ckpt_path = Path(checkpoint_dir) / 't5-finetuned'
    model.save_pretrained(ckpt_path)
    print(f'Model checkpoints saved at: {ckpt_path}')


def finetune_gpt2(
    variant: str = 'gpt2',
    device: torch.device = 'cpu',
    checkpoint_dir: str = 'checkpoints',
    dataset_dir: str = 'repositories',
    batch_size: int = 16,
    use_fp16: bool = True,
):
    # instantiate pretrained tokenizer and model
    model = GPT2LMHeadModel.from_pretrained(variant)
    tokenizer = GPT2TokenizerFast.from_pretrained(variant)

    # put model on cuda device and set it to training mode
    model.to(device)
    model.train()

    # use half-precision format during training
    # which leads to shorter training time and lower memory requirements
    # also enabling larger batch sizes
    if use_fp16:
        model.half()

    # Padding tokens were not used during the pre-training of GPT and GPT-2, therefore they have none.
    # An attention mask should be specified so that the model won't attend to padded indices.
    # A padding token is set here anyway because we want the tokenizer to return tensors.
    tokenizer.pad_token = tokenizer.eos_token

    # retrieve python repositories dataset
    block_size = 2 << 7
    dataset = PythonReposDataset(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        count_fn=lambda x: len(tokenizer(x)['input_ids']),
        block_size=block_size,
    )

    # initialize model's optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # The goal of this finetuning is to let the model see each of the python source
    # files exactly once (and not by epochs)
    for i, batch in enumerate(dataset):
        # encode batch into their token IDS
        encoding = tokenizer(batch, return_tensors='pt', padding=True)
        input_ids = encoding['input_ids']
        attn_mask = encoding['attention_mask']

        # split tensors since the model has a max length limit
        input_ids = input_ids.transpose(0, 1).split(block_size)
        attn_mask = attn_mask.transpose(0, 1).split(block_size)

        for j in range(len(input_ids)):
            _input_ids = input_ids[j].transpose(0, 1).to(device)
            _attn_mask = attn_mask[j].transpose(0, 1).to(device)

            # compute loss
            loss, _, _ = model(
                _input_ids,
                attention_mask=_attn_mask,
                labels=_input_ids,
            )
            print(f'Step {i+1}-{j+1}: Loss={loss}')

            # delete input tensors to free memory in the GPU
            del _input_ids, _attn_mask

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save finetuned weights
    ckpt_path = Path(checkpoint_dir) / 'gpt2-finetuned'
    model.save_pretrained(ckpt_path)
    print(f'Model checkpoints saved at: {ckpt_path}')


if __name__ == '__main__':
    # set random seed for reproducibility
    torch.manual_seed(7)

    # check for CUDA availability
    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available. Training will be slow.")

    fire.Fire(finetune)
