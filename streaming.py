"""
Switching a model between whole-signal and chunk-at-a-time operation.

    with streaming(encoder):
        for chunk in chunks:
            codes = encoder(chunk)

Inside the block each submodule carries whatever it needs between calls,
so the concatenated result matches processing the signal in one go. On
exit the state is dropped and the model is offline again.
"""

from contextlib import contextmanager


def reset_stream(model, streaming: bool = True):
    """Clear every submodule's carried state and set the mode."""
    for module in model.modules():
        reset = getattr(module, "reset_stream_state", None)

        if reset is not None:
            reset(streaming)


@contextmanager
def streaming(*models):
    """Run a block with the given models in streaming mode."""
    for model in models:
        reset_stream(model, True)

    try:
        yield
    finally:
        for model in models:
            reset_stream(model, False)
