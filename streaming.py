from contextlib import contextmanager

from torch import nn


class StreamingModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.streaming = False

    def reset_stream_state(self, streaming: bool):
        self.streaming = streaming


def reset_stream(model, streaming: bool = True):
    for module in model.modules():
        reset = getattr(module, "reset_stream_state", None)

        if reset is not None:
            reset(streaming)


@contextmanager
def streaming(*models):
    for model in models:
        reset_stream(model, True)

    try:
        yield
    finally:
        for model in models:
            reset_stream(model, False)
