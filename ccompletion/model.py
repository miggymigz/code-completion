from transformers import TFGPT2LMHeadModel, GPT2Config


def batch_attention_mask(batch):
    lengths = [len(sample) for sample in batch]
    max_length = max(lengths)
    mask = []

    for l in lengths:
        sample_mask = [1] * l + [0] * (max_length - l)
        mask.append(sample_mask)

    return mask


def create_model(*, n_vocab=24_000):
    # Initialize GPT-2 configuration
    config = GPT2Config(
        vocab_size=n_vocab,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=6,
        n_head=12,
        activation_function='gelu_new',
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-05,
        initializer_range=0.02,
    )

    return TFGPT2LMHeadModel(config)
