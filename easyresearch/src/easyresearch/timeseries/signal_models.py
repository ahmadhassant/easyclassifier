"""Classifiers for recordings (one label per recording), default settings.

All are scikit-learn style estimators taking a table with one recording per
row (time points as columns), so EasyClassifier's evaluation (cross-
validation, nested cross-validation, final test set, measures and figures)
is reused unchanged. Per-recording scaling is done inside each estimator and
uses only that recording, so no information passes between recordings.

* Most frequent class      - reference rule
* KNN, Hassanat distance   - each recording scaled to 0-1, k = 5
* ROCKET                   - 10,000 random convolution kernels and a ridge
                             classifier (Dempster, Petitjean & Webb, 2020)
* Signal features + Random Forest - 20 summary features per recording
* Deep: FCN and ResNet (Wang, Yan & Oates, 2017), InceptionTime (one
  network; Ismail Fawaz et al., 2020) and LSTM; each recording
  standardised (mean 0, SD 1)
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch import nn
import torch.nn.functional as F

from easyclassifier.distances import HassanatKNN

from ..common.references import REFERENCES, SIGNAL_MODEL_REFERENCES
from . import deep
from .deep import device

SEED = 0
warnings.filterwarnings("ignore", message="Using padding='same'")


# --------------------------------------------------------------------------- #
# Per-recording scaling (stateless)
# --------------------------------------------------------------------------- #

def _arr(X):
    return np.asarray(X, dtype=float)


def znorm(X):
    X = _arr(X)
    sd = X.std(axis=1, keepdims=True)
    return (X - X.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1)


def minmax(X):
    X = _arr(X)
    lo, hi = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    return (X - lo) / np.where(hi > lo, hi - lo, 1)


# --------------------------------------------------------------------------- #
# Summary features
# --------------------------------------------------------------------------- #

FEATURE_NAMES = ["mean", "sd", "min", "max", "median", "q25", "q75",
                 "skewness", "kurtosis", "slope", "autocorr_1",
                 "autocorr_5", "mean_crossings", "argmax_pos", "argmin_pos",
                 "energy_band_1", "energy_band_2", "energy_band_3",
                 "energy_band_4", "mean_abs_change"]


def features(X):
    X = _arr(X)
    n, L = X.shape
    m, sd = X.mean(1), X.std(1)
    z = znorm(X)
    t = np.arange(L) - (L - 1) / 2
    slope = (X * t).sum(1) / (t ** 2).sum()

    def ac(k):
        return (z[:, :-k] * z[:, k:]).mean(1) if L > k else np.zeros(n)
    spec = np.abs(np.fft.rfft(z, axis=1)) ** 2
    bands = np.array_split(spec[:, 1:], 4, axis=1)
    energy = np.stack([b.sum(1) for b in bands], 1)
    energy = energy / np.maximum(energy.sum(1, keepdims=True), 1e-12)
    return np.column_stack([
        m, sd, X.min(1), X.max(1), np.median(X, 1),
        np.percentile(X, 25, 1), np.percentile(X, 75, 1),
        (z ** 3).mean(1), (z ** 4).mean(1) - 3, slope, ac(1), ac(5),
        (np.diff(np.sign(z), axis=1) != 0).mean(1),
        X.argmax(1) / L, X.argmin(1) / L, energy,
        np.abs(np.diff(X, axis=1)).mean(1)])


# --------------------------------------------------------------------------- #
# ROCKET
# --------------------------------------------------------------------------- #

class Rocket(BaseEstimator, ClassifierMixin):
    """Random convolutional kernels (Dempster et al., 2020): for each
    kernel the share of positive values and the maximum of the convolution
    are features for a ridge classifier."""

    def __init__(self, n_kernels: int = 10000, random_state: int = SEED):
        self.n_kernels = n_kernels
        self.random_state = random_state

    def _make_kernels(self, L):
        rng = np.random.default_rng(self.random_state)
        groups = {}
        for _ in range(self.n_kernels):
            length = int(rng.choice([7, 9, 11]))
            w = rng.normal(0, 1, length)
            w -= w.mean()
            b = rng.uniform(-1, 1)
            max_exp = np.log2(max(1, (L - 1) / (length - 1)))
            dil = int(2 ** rng.uniform(0, max_exp))
            pad = ((length - 1) * dil) // 2 if rng.integers(2) else 0
            groups.setdefault((length, dil, pad), []).append((w, b))
        self.groups_ = [(k, torch.tensor(np.stack([w for w, _ in v])[:, None],
                                         dtype=torch.float32),
                         torch.tensor([b for _, b in v], dtype=torch.float32))
                        for k, v in groups.items()]

    def _transform(self, X):
        x = torch.tensor(znorm(X), dtype=torch.float32)[:, None, :]
        feats = []
        with torch.no_grad():
            for (length, dil, pad), W, b in self.groups_:
                parts = []
                for s in range(0, len(x), 256):
                    out = F.conv1d(x[s:s + 256], W, dilation=dil,
                                   padding=pad) + b[None, :, None]
                    parts.append(torch.cat([(out > 0).float().mean(2),
                                            out.amax(2)], 1))
                feats.append(torch.cat(parts, 0))
        return torch.cat(feats, 1).numpy()

    def fit(self, X, y):
        X = _arr(X)
        self._make_kernels(X.shape[1])
        self.clf_ = make_pipeline(StandardScaler(),
                                  RidgeClassifierCV(np.logspace(-3, 3, 10)))
        self.clf_.fit(self._transform(X), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(self._transform(X))

    def predict_proba(self, X):
        d = self.clf_.decision_function(self._transform(X))
        if d.ndim == 1:
            d = np.column_stack([-d, d])
        e = np.exp(d - d.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


# --------------------------------------------------------------------------- #
# Deep networks
# --------------------------------------------------------------------------- #

def _conv_bn(cin, cout, k):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding="same"),
                         nn.BatchNorm1d(cout), nn.ReLU())


class FCN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.body = nn.Sequential(_conv_bn(1, 128, 8), _conv_bn(128, 256, 5),
                                  _conv_bn(256, 128, 3))
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.body(x.unsqueeze(1)).mean(2))


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.convs = nn.Sequential(
            _conv_bn(cin, cout, 8), _conv_bn(cout, cout, 5),
            nn.Conv1d(cout, cout, 3, padding="same"), nn.BatchNorm1d(cout))
        self.short = nn.Sequential(nn.Conv1d(cin, cout, 1),
                                   nn.BatchNorm1d(cout))

    def forward(self, x):
        return F.relu(self.convs(x) + self.short(x))


class ResNet1D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.body = nn.Sequential(ResBlock(1, 64), ResBlock(64, 128),
                                  ResBlock(128, 128))
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.body(x.unsqueeze(1)).mean(2))


class Inception(nn.Module):
    def __init__(self, cin, nf=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(cin, nf, 1, bias=False) if cin > 1 \
            else nn.Identity()
        cb = nf if cin > 1 else 1
        self.convs = nn.ModuleList(nn.Conv1d(cb, nf, k, padding="same",
                                             bias=False) for k in (39, 19, 9))
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1),
                                  nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(4 * nf)

    def forward(self, x):
        b = self.bottleneck(x)
        return F.relu(self.bn(torch.cat([c(b) for c in self.convs]
                                        + [self.pool(x)], 1)))


class InceptionNet(nn.Module):
    def __init__(self, n_classes, depth=6, nf=32):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.shorts = nn.ModuleList()
        cin, res = 1, 1
        for d in range(depth):
            self.blocks.append(Inception(cin, nf))
            cin = 4 * nf
            if d % 3 == 2:
                self.shorts.append(nn.Sequential(
                    nn.Conv1d(res, cin, 1, bias=False), nn.BatchNorm1d(cin)))
                res = cin
        self.head = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        res = x
        for d, block in enumerate(self.blocks):
            x = block(x)
            if d % 3 == 2:
                x = F.relu(x + self.shorts[d // 3](res))
                res = x
        return self.head(x.mean(2))


class LSTMClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.rnn = nn.LSTM(1, 64, batch_first=True)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.head(out[:, -1])


NETWORKS = {"fcn": FCN, "resnet": ResNet1D, "inception": InceptionNet,
            "lstm": LSTMClassifier}


class DeepClassifier(BaseEstimator, ClassifierMixin):
    """Adam (learning rate 0.001), batches of 16, cross-entropy, at most 150
    epochs; stops after 20 epochs without improvement on a stratified 10%
    of the training recordings and keeps the best weights."""

    MAX_EPOCHS, PATIENCE, BATCH, LR = 150, 20, 16, 1e-3

    def __init__(self, arch: str = "fcn", random_state: int = SEED):
        self.arch = arch
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = znorm(X)
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        dev = device()
        net = NETWORKS[self.arch](len(self.classes_)).to(dev)
        idx = np.arange(len(X))
        counts = np.bincount(yi)
        if len(X) >= 40 and counts.min() >= 2:
            tr, va = train_test_split(idx, test_size=0.1, stratify=yi,
                                      random_state=self.random_state)
        else:
            tr, va = idx, idx
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        yt = torch.tensor(yi, dtype=torch.long, device=dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.LR)
        gen = torch.Generator().manual_seed(self.random_state)
        best, best_loss, waited = copy.deepcopy(net.state_dict()), np.inf, 0
        tr_t = torch.tensor(tr)
        for _ in range(self.MAX_EPOCHS):
            net.train()
            order = tr_t[torch.randperm(len(tr), generator=gen)]
            for s in range(0, len(order), self.BATCH):
                b = order[s:s + self.BATCH]
                if len(b) < 2:          # batch norm needs two recordings
                    continue
                b = b.to(dev)
                opt.zero_grad()
                F.cross_entropy(net(Xt[b]), yt[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(F.cross_entropy(net(Xt[va]), yt[va]))
            deep.epoch_done()
            if v < best_loss - 1e-6:
                best, best_loss, waited = copy.deepcopy(net.state_dict()), v, 0
            else:
                waited += 1
                if waited >= self.PATIENCE:
                    break
        net.load_state_dict(best)
        self.net_ = net.cpu().eval()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            logits = self.net_(torch.tensor(znorm(X), dtype=torch.float32))
            return torch.softmax(logits, 1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# --------------------------------------------------------------------------- #
# Registry (EasyClassifier-compatible specs)
# --------------------------------------------------------------------------- #

@dataclass
class SignalSpec:
    key: str
    name: str
    factory: Callable[[], object]
    family: str
    default_selected: bool = False
    available: bool = True
    reason: str = ""
    help_key: str = ""


def build_registry() -> Dict[str, SignalSpec]:
    specs = [
        SignalSpec("majority", "Reference: most frequent class",
                   lambda: DummyClassifier(strategy="most_frequent"),
                   "reference"),
        SignalSpec("knn", "KNN (Hassanat distance, k=5)",
                   lambda: make_pipeline(FunctionTransformer(minmax),
                                         HassanatKNN(5)), "distance",
                   default_selected=True),
        SignalSpec("rocket", "ROCKET (random convolution kernels)", Rocket,
                   "features", default_selected=True),
        SignalSpec("features_rf", "Signal features + Random Forest",
                   lambda: make_pipeline(FunctionTransformer(features),
                                         RandomForestClassifier(
                                             random_state=SEED)),
                   "features", default_selected=True),
        SignalSpec("fcn", "Deep: FCN", lambda: DeepClassifier("fcn"), "deep"),
        SignalSpec("resnet", "Deep: ResNet", lambda: DeepClassifier("resnet"),
                   "deep"),
        SignalSpec("inception", "Deep: InceptionTime (one network)",
                   lambda: DeepClassifier("inception"), "deep"),
        SignalSpec("lstm", "Deep: LSTM", lambda: DeepClassifier("lstm"),
                   "deep"),
    ]
    return {s.key: s for s in specs}


# Citation texts live in common/references.py (one list for every report);
# the first reference of each model, kept for compatibility.
CITATIONS = {k: REFERENCES[v[0]] for k, v in SIGNAL_MODEL_REFERENCES.items()
             if v and k in ("rocket", "fcn", "resnet", "inception", "lstm")}
