from ccompletion.dataset import PythonRepositoriesDataset
from ccompletion.hparams import LRCustomSchedule, get_hparams
from ccompletion.layers.attention import create_masks
from ccompletion.model import build_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

import fire
import json
import os
import time
import torch


def train(epochs=1000, batch_size=16, max_to_keep=5,
          checkpoint_dir='checkpoints/train', start_from_zero=False):
    dataset = PythonRepositoriesDataset(saved_file='dataset.pt')
    hparams = get_hparams()
    print(hparams)

    # build model archi based on hparams
    model = build_model(**hparams)

    # model.create_optimizer()
    # model.create_checkpoint_manager(
    #     checkpoint_path=checkpoint_dir,
    #     max_to_keep=max_to_keep,
    #     load_model=not start_from_zero,
    # )
    # model.fit(dataset, epochs=epochs)

    # data_loader = init_data_loader(batch_size=batch_size)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=0.0001,
    #     betas=(0.9, 0.999),
    #     eps=1e-9,
    # )

    # for epoch in range(epochs):
    #     acc_loss = 0
    #     for i, data in enumerate(data_loader):
    #         inputs, targets = data
    #         optimizer.zero_grad()

    #         outputs, _ = model(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()

    #         # print training stats
    #         acc_loss += loss.item()
    #         if i % 100 == 99:
    #             print('[%d, %4d] loss: %.3f' % (epoch+1, i+1, acc_loss / 100))
    #             acc_loss = 0

    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=True,
    #     mlm_probability=0.15,
    # )

    training_args = TrainingArguments(
        output_dir='code-completion-output',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train()


def init_data_loader(*, batch_size):
    dataset = PythonRepositoriesDataset(saved_file='dataset.pt')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    return loader


if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    if n_gpus:
        print('INFO - %d GPUs detected. Will use GPU to train model.' % n_gpus)
    else:
        print('WARNING - No GPU was detected.')
        print('WARNING - Training will take a very long time.')

    fire.Fire(train)
