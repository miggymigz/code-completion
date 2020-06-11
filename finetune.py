from ccompletion.torch_dataset import MyDataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AdamW, get_linear_schedule_with_warmup

import fire
import os
import torch


def finetune(
    epochs=5,
    batch_size=16,
    learning_rate=3e-5,
    warmup_steps=5000,
    max_seq_len=400,
    output_dir='trained_models',
):
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    dataset = MyDataset(dataset_dir='repositories')
    loader = DataLoader(dataset, batch_size=1)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=-1,
    )

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    proc_seq_count = 0
    batch_count = 0
    sum_loss = 0

    for epoch in range(epochs):
        print(f'[INFO] EPOCH {epoch+1} started')

        for _, code in enumerate(loader):
            outputs = model(code, labels=code)
            loss, logits = outputs[:2]
            loss.backward()
            sum_loss += loss.detach().data

            proc_seq_count += 1
            if proc_seq_count == batch_size:
                proc_seq_count = 0
                batch_count += 0

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 100:
                print(f'sum loss {sum_loss}')
                batch_count = sum_loss = 0

        ckpt_path = os.path.join(output_dir, f'gpt2_{epoch}.pt')
        torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    fire.Fire(finetune)
