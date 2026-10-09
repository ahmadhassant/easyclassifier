"""Deep forecasting networks (PyTorch) and their training.

Every network maps the last L values (a window) to the next H values at
once. Defaults are fixed, not tuned: Adam with learning rate 0.001, batches
of 32, mean squared error, at most 200 passes over the data, stopping when
the error on the most recent 10% of training windows has not improved for
20 passes, and keeping the best weights. Seeds are fixed, so a run on the
same computer gives the same result.

Architectures (kept small, as suits the few hundred to few thousand values
of a typical research series):

* MLP          - two hidden layers of 128 units
* LSTM / GRU   - one recurrent layer of 64 units (Hochreiter & Schmidhuber
                 1997; Cho et al. 2014)
* TCN          - causal dilated 1-D convolutions, 32 channels, dilations
                 1, 2, 4, 8 with residual connections (Bai et al. 2018)
* N-BEATS      - three generic blocks of four 128-unit layers with backcast
                 and forecast outputs (Oreshkin et al. 2020)
* Transformer  - two encoder layers, model width 32, four attention heads
                 (Vaswani et al. 2017)
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..common.references import FORECAST_MODEL_REFERENCES, REFERENCES

SEED = 0
MAX_EPOCHS = 200

# Called after every training epoch when set (the desktop application uses it
# to move its progress bar during long trainings); it never affects training.
EPOCH_HOOK = None


def epoch_done() -> None:
    if EPOCH_HOOK is not None:
        EPOCH_HOOK()
PATIENCE = 20
BATCH = 32
LR = 1e-3

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, L, H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(L, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, H))

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    def __init__(self, L, H, cell="lstm"):
        super().__init__()
        layer = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = layer(1, 64, batch_first=True)
        self.head = nn.Linear(64, H)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.head(out[:, -1])


class TCN(nn.Module):
    def __init__(self, L, H, ch=32, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.dilations = dilations
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        cin = 1
        for d in dilations:
            self.convs.append(nn.Conv1d(cin, ch, 3, dilation=d))
            self.skips.append(nn.Conv1d(cin, ch, 1) if cin != ch
                              else nn.Identity())
            cin = ch
        self.head = nn.Linear(ch, H)

    def forward(self, x):
        h = x.unsqueeze(1)
        for conv, skip, d in zip(self.convs, self.skips, self.dilations):
            out = F.relu(conv(F.pad(h, (2 * d, 0))))     # causal padding
            h = out + skip(h)
        return self.head(h[:, :, -1])


class NBeatsBlock(nn.Module):
    def __init__(self, L, H, width=128, layers=4):
        super().__init__()
        mods, cin = [], L
        for _ in range(layers):
            mods += [nn.Linear(cin, width), nn.ReLU()]
            cin = width
        self.fc = nn.Sequential(*mods)
        self.back = nn.Linear(width, L)
        self.fore = nn.Linear(width, H)

    def forward(self, x):
        h = self.fc(x)
        return self.back(h), self.fore(h)


class NBeats(nn.Module):
    def __init__(self, L, H, blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList(NBeatsBlock(L, H) for _ in range(blocks))

    def forward(self, x):
        forecast = 0
        for block in self.blocks:
            back, fore = block(x)
            x = x - back
            forecast = forecast + fore
        return forecast


class Transformer(nn.Module):
    def __init__(self, L, H, d=32):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        layer = nn.TransformerEncoderLayer(d, 4, 64, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(L * d, H)

    def forward(self, x):
        h = self.embed(x.unsqueeze(-1)) + self.pos
        return self.head(self.encoder(h).flatten(1))


ARCHITECTURES = {
    "mlp": lambda L, H: MLP(L, H),
    "lstm": lambda L, H: RNN(L, H, "lstm"),
    "gru": lambda L, H: RNN(L, H, "gru"),
    "tcn": lambda L, H: TCN(L, H),
    "nbeats": lambda L, H: NBeats(L, H),
    "transformer": lambda L, H: Transformer(L, H),
}


def train(arch: str, X: np.ndarray, Y: np.ndarray, seed: int = SEED):
    """Train a network on windows X (n, L) -> targets Y (n, H)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    L, H = X.shape[1], Y.shape[1]
    net = ARCHITECTURES[arch](L, H).to(dev)
    n_val = max(1, int(round(0.1 * len(X)))) if len(X) >= 30 else 0
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    Xtr, Ytr = Xt[:len(X) - n_val], Yt[:len(X) - n_val]
    Xva, Yva = Xt[len(X) - n_val:], Yt[len(X) - n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_loss, waited = copy.deepcopy(net.state_dict()), np.inf, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        order = torch.randperm(len(Xtr), generator=gen)
        for s in range(0, len(Xtr), BATCH):
            idx = order[s:s + BATCH].to(dev)
            opt.zero_grad()
            loss = F.mse_loss(net(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            Xe, Ye = (Xva, Yva) if n_val else (Xtr, Ytr)
            val = float(F.mse_loss(net(Xe), Ye))
        epoch_done()
        if val < best_loss - 1e-7:
            best, best_loss, waited = copy.deepcopy(net.state_dict()), val, 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    net.load_state_dict(best)
    net.eval()
    return net.cpu()


def predict(net, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net(torch.tensor(x, dtype=torch.float32)).numpy()


# Citation texts live in common/references.py (one list for every report).
DEEP_CITATIONS = {k: REFERENCES[v[0]] for k, v in
                  FORECAST_MODEL_REFERENCES.items()
                  if k in ("lstm", "gru", "tcn", "nbeats", "transformer")}
PYTORCH_CITATION = REFERENCES["pytorch"]
