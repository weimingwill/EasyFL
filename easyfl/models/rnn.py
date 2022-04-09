import torch
import torch.nn as nn

from easyfl.models.model import BaseModel


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class Model(BaseModel):
    def __init__(self, embedding_dim=8, voc_size=80, lstm_unit=256, batch_first=True, n_layers=2):
        super(Model, self).__init__()
        self.encoder = nn.Embedding(voc_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_unit, n_layers, batch_first=batch_first)
        self.decoder = nn.Linear(lstm_unit, voc_size)
        self.init_weights()

    def forward(self, inp):
        inp = self.encoder(inp)
        inp, _ = self.lstm(inp)
        # extract the last state of output for prediction
        hidden = inp[:, -1]
        output = self.decoder(hidden)
        return output

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
