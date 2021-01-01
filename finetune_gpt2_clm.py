from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel,
    Trainer, TrainingArguments, default_data_collator,
)

import fire
import os


def finetune_gpt2_clm(variant: str = 'gpt2-medium'):
    # initialize huggingface datasets
    datasets = load_dataset('load_dataset_script.py', data_dir='repositories')

    # initialize model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(variant)
    tokenizer = GPT2TokenizerFast.from_pretrained(variant)
    block_size = tokenizer.model_max_length

    # preprocessing datasets
    tokenized_datasets = datasets.map(
        lambda sample: tokenizer(sample['src']),
        batched=True,
        num_proc=10,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together,
    # so group_texts throws away a remainder for each of those groups of 1,000 texts.
    # You can adjust that batch_size here but a higher value might be slower to preprocess.
    # To speed up this part, we use multiprocessing.
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=10,
    )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir='finetuned_gpt2_clm',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        do_predict=False,
        num_train_epochs=1,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"] if training_args.do_train else None,
        eval_dataset=lm_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )

    # Training
    train_result = trainer.train(model_path=None)
    trainer.save_model()
    output_train_file = os.path.join(
        training_args.output_dir,
        "train_results.txt"
    )

    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            for key, value in sorted(train_result.metrics.items()):
                writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer_state_path = os.path.join(
                training_args.output_dir, "trainer_state.json")
            trainer.state.save_to_json(trainer_state_path)


if __name__ == "__main__":
    fire.Fire(finetune_gpt2_clm)
