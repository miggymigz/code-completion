import torch
from transformers import data
from dataset import PythonReposDataset
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AdamW

import fire
import math


def finetune(
    gpt2_variant: str = 'gpt2',
    checkpoint_dir: str = 'checkpoints',
    dataset_dir: str = 'repositories',
    batch_size: int = 8
):
    # instantiate pretrained tokenizer and model
    model = GPT2LMHeadModel.from_pretrained(gpt2_variant)
    tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_variant)

    # set model to training mode
    model.train()

    # Padding tokens were not used during the pre-training of GPT and GPT-2, therefore they have none.
    # An attention mask should be specified so that the model won't attend to padded indices.
    # A padding token is set here anyway because we want the tokenizer to return tensors.
    tokenizer.pad_token = tokenizer.eos_token

    # retrieve python repositories dataset
    n_positions = model.config.n_positions
    dataset = PythonReposDataset(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        count_fn=lambda x: len(tokenizer(x)['input_ids']),
        max_position=n_positions,
    )

    # initialize model's optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # The goal of this finetuning is to let the model see each of the python source
    # files exactly once (and not by epochs)
    for i, batch in enumerate(dataset):
        # encode batch into their token IDS
        encoding = tokenizer(
            batch, return_tensors='pt',
            padding=True
        )
        input_ids = encoding['input_ids']
        attn_mask = encoding['attention_mask']

        # split tensors since the model has a max length limit
        input_ids = torch.split(input_ids.transpose(0, 1), n_positions)
        attn_mask = torch.split(attn_mask.transpose(0, 1), n_positions)
        past = None

        for j in range(len(input_ids)):
            _input_ids = input_ids[j].transpose(0, 1)
            _attn_mask = attn_mask[j].transpose(0, 1)

            # compute loss
            loss, _, past = model(
                _input_ids,
                attention_mask=_attn_mask,
                past_key_values=past,
                labels=_input_ids,
            )
            print(f'Batch {i+1:3d}: Loss={loss}')

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save finetuned weights
    ckpt_path = Path(checkpoint_dir) / 'gpt2-finetuned'
    model.save_pretrained(ckpt_path)
    print(f'Model checkpoints saved at: {ckpt_path}')


if __name__ == '__main__':
    fire.Fire(finetune)
