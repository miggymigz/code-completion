from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AdamW

import fire
import glob
import shutil
import torch


def finetune_t5(
    variant: str = 't5-base',
    learning_rate: float = 1e-4,
    device: torch.device = 'cpu',
    dataset_dir: str = 'repositories',
    batch_size: int = 8,
    use_fp16: bool = True,
    steps_per_checkpoint: int = 100,
    ckpt_path: str = 't5-finetuned',
):
    # load dataset specially configured for T5 finetuning
    dataset = load_dataset(
        'load_python_repos.py',
        data_dir=dataset_dir,
        clean=True,
        model='t5',
        split='train',
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # instantiate model from the checkpoint (if exists), else variant
    if Path(ckpt_path).is_dir():
        print(f'Will load from checkpoint "{ckpt_path}".')
        model = T5ForConditionalGeneration.from_pretrained(ckpt_path)
    else:
        print(f'Will start finetuning from scratch.')
        model = T5ForConditionalGeneration.from_pretrained(variant)

    # put model on cuda device and set it to training mode
    model.to(device)
    model.train()

    # use half-precision format during training
    # which leads to shorter training time and lower memory requirements
    # also enabling larger batch sizes
    if torch.cuda.is_available() and use_fp16:
        model.half()

    # instantiate tokenizer based on the passed variant
    tokenizer = T5TokenizerFast.from_pretrained(variant)

    # initialize model's optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # initialize tqdm progress bar (with optional offset)
    pbar = tqdm(dataloader)

    # The goal of this finetuning is to let the model see each of the python source
    # files exactly once (and not by epochs)
    for i, batch in enumerate(pbar):
        inputs = tokenizer(batch['src'], padding=True, return_tensors='pt')
        labels = tokenizer(batch['target'], padding=True, return_tensors='pt')
        inputs['labels'] = labels.input_ids + \
            ((labels.attention_mask - 1) * 100)  # -100 are masked

        # move input tensor to appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # compute loss
        loss = model(**inputs).loss

        # if loss turns out to be nan, then there's something wrong
        # with the inputs that was fed into the model so training should not continue
        if torch.isnan(loss):
            model.save_pretrained(f'{ckpt_path}-stopped-by-nan')
            pbar.write('Input batch that lead to nan loss:')
            pbar.write(str(dataset[i]))
            return

        # Write loss and continue training
        pbar.set_description(f'Batch#{i}: Loss={loss}')

        # delete input tensors to free memory in the GPU
        del inputs

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


if __name__ == '__main__':
    fire.Fire(finetune_t5)
