from transformers import TFGPT2LMHeadModel, GPT2Config


def create_model(*, n_vocab):
    # Initialize GPT-2 configuration
    config = GPT2Config(
        vocab_size=n_vocab,
        n_ctx=1024,
        n_positions=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        activation_function='gelu_new',
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-05,
        initializer_range=0.02,
    )

    return TFGPT2LMHeadModel(config)
