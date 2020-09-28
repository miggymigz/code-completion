from torch.utils.data.dataloader import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AdamW
from tqdm import tqdm

from dataset import PythonReposDataset

import fire
import math


def finetune(variant: str = 'gpt2', dataset_dir: str = 'repositories', batch_size: int = 8):
    # instantiate pretrained tokenizer and model
    model = GPT2LMHeadModel.from_pretrained(variant)
    tokenizer = GPT2TokenizerFast.from_pretrained(variant)

    # set model to training mode
    model.train()

    # Padding tokens were not used during the pre-training of GPT and GPT-2, therefore they have none.
    # An attention mask should be specified so that the model won't attend to padded indices.
    # A padding token is set here anyway because we want the tokenizer to return tensors.
    tokenizer.pad_token = tokenizer.eos_token

    # retrieve python repositories dataset
    dataset = PythonReposDataset(dataset_dir=dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # The goal of this finetuning is to let the model see each of the python source
    # files exactly once (and not by epochs)
    n_total_steps = math.ceil(len(dataset) / batch_size)
    pbar = tqdm(enumerate(dataloader), total=n_total_steps)
    for i, batch in pbar:
        # encode batch into their token IDS
        encoding = tokenizer(
            batch, return_tensors='pt',
            padding=True, truncation=True,
        )
        input_ids = encoding['input_ids']
        attn_mask = encoding['attention_mask']

        # compute loss
        optimizer = AdamW(model.parameters(), lr=1e-5)
        loss = model(input_ids, attention_mask=attn_mask, labels=input_ids)[0]
        pbar.write(f'Batch {i:3d}: Loss={loss}')

        # update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    fire.Fire(finetune)
