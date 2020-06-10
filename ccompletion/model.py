from transformers import GPT2LMHeadModel, GPT2Config


def build_model(*, n_vocab, n_ctx, n_embd, n_head, n_layer):
    # Initialize GPT-2 configuration
    config = GPT2Config(
        vocab_size=n_vocab,
        n_positions=n_ctx,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        activation_function='gelu',
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=16,
    )

    return GPT2LMHeadModel(config)

