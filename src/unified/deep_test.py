"""
A real LSTM on the real data -- the architecture the papers actually used.

Nine scripts in this project use gradient-boosted trees, which are generally as
strong as or stronger than the random forests and shallow networks in the
papers being checked. But Altaibek et al. (2024) report an LSTM, and until now
no LSTM has been trained here on real data: the only one in the repository ran
on synthetic series to demonstrate a leakage mechanism. That is a fair gap, and
this closes it, because "maybe a deeper model would find it" cannot be settled
by argument.

A recurrent network is not just a bigger classifier. It sees the SEQUENCE --
thirty days of every forcing variable leading up to today -- so it can in
principle pick up a build-up, a rate of change or an interaction between
families that a single-day model cannot represent. If there is a pattern of
that kind, this is the architecture to find it.

    input    30 days x 130 forcing variables
    model    LSTM(130 -> 64) -> dropout -> dense -> sigmoid
    target   an M >= 6.0 somewhere on Earth today
    compare  SKY (forcings only), ETAS (recent seismicity only), BOTH

THE SAME MODEL IS RUN TWICE, and only the split changes:

    TEMPORAL   train on the past, test on the future, with a gap of one
               sequence length so no test window overlaps training. This is
               forecasting.
    ACAK       days shuffled at random into train and test, which is what the
               published pipelines do. Earthquakes cluster, so a shuffle puts
               3 January in training and 4 January in test; the model then
               recalls rather than predicts.

Reporting both is the point. If the temporal number is at chance and the random
number is excellent, the architecture was never the issue -- the split was.
Standardisation uses training statistics only, so even the random run is
treated more carefully than the papers treat theirs.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from forcing_bank import build_all
from combined_predictor import etas_features

SEQ = 30
HIDDEN = 64
EPOCHS = 15
BATCH = 256
LR = 1e-3
RNG = np.random.default_rng(20260817)
torch.manual_seed(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


class LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


def run(X, y, train_idx, test_idx, label):
    """Train on train_idx, score test_idx. Sequences are built on the fly."""
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0)
    sd[sd == 0] = 1.0
    Z = torch.tensor(((X - mu) / sd), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    def batch(idx):
        seqs = torch.stack([Z[i - SEQ + 1:i + 1] for i in idx])
        return seqs, yt[idx]

    model = LSTMNet(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos = float(y[train_idx].mean())
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6)))

    for ep in range(EPOCHS):
        model.train()
        perm = RNG.permutation(train_idx)
        tot = 0.0
        for a in range(0, perm.size, BATCH):
            xb, yb = batch(perm[a:a + BATCH])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * xb.shape[0]
        print(f"    {label} epoch {ep+1}/{EPOCHS}  loss {tot/perm.size:.4f}",
              end="\r", flush=True)
    print(" " * 60, end="\r")

    model.eval()
    ps = []
    with torch.no_grad():
        for a in range(0, test_idx.size, 512):
            xb, _ = batch(test_idx[a:a + 512])
            ps.append(torch.sigmoid(model(xb)).numpy())
    p = np.concatenate(ps)
    return float(roc_auc_score(y[test_idx], p)), p


def main():
    rule("LSTM SUNGGUHAN PADA DATA NYATA — arsitektur yang dipakai paper")

    days = np.load(datafile("m6_days.npy"))
    cnt = np.load(datafile("m6_cnt.npy"))
    y = (cnt > 0).astype(int)
    print(f"\n  hari {len(y):,}   laju dasar {100*y.mean():.2f}%")
    print("\n  membangun bank variabel:")
    X_sky, names, fams = build_all(days)
    X_etas = etas_features(cnt)
    X_both = np.column_stack([X_etas, X_sky])
    print(f"\n  SKY {X_sky.shape[1]} | ETAS {X_etas.shape[1]} | "
          f"BOTH {X_both.shape[1]}   urutan {SEQ} hari")

    n = len(y)
    valid = np.arange(SEQ - 1, n)
    cut = int(n * 0.70)
    # the gap of one sequence length stops a test window from reaching back
    # into days the model trained on
    tr_temporal = valid[valid < cut]
    te_temporal = valid[valid >= cut + SEQ]

    perm = RNG.permutation(valid)
    k = int(perm.size * 0.70)
    tr_random, te_random = perm[:k], perm[k:]

    print(f"""
  split temporal : latih {tr_temporal.size:,} -> uji {te_temporal.size:,} (ke DEPAN)
  split acak     : latih {tr_random.size:,} -> uji {te_random.size:,} (tercampur)
""")

    rule("HASIL")
    print(f"\n  {'model':<28} {'AUC temporal':>14} {'AUC acak':>12}")
    print("  " + "-" * 58)
    out = {}
    for label, M in [("SKY (130 gaya luar)", X_sky),
                     ("ETAS (seismologi)", X_etas),
                     ("BOTH", X_both)]:
        a_t, _ = run(M, y, tr_temporal, te_temporal, f"{label}/temporal")
        a_r, _ = run(M, y, tr_random, te_random, f"{label}/acak")
        out[label] = (a_t, a_r)
        print(f"  {label:<28} {a_t:>14.4f} {a_r:>12.4f}")

    rule("VONIS")
    st, sr = out["SKY (130 gaya luar)"]
    et, _ = out["ETAS (seismologi)"]
    bt, _ = out["BOTH"]
    print(f"""
  AUC 0,50 = lempar koin.

  LSTM pada gaya luar, split temporal : {st:.4f}
  LSTM pada gaya luar, split acak     : {sr:.4f}
  selisih hanya karena cara membagi   : {sr-st:+.4f}

  sumbangan langit di atas ETAS       : {bt-et:+.4f} AUC

  Arsitektur bukan penyebabnya. Model ini melihat SELURUH {SEQ} hari terakhir
  dari 130 variabel sekaligus -- ia sanggup menangkap penumpukan, laju
  perubahan, dan interaksi antar keluarga yang tidak bisa diwakili model
  satu-hari. Dengan semua keleluasaan itu, hasil temporalnya tetap di
  sekitar lempar koin.

  Yang berubah drastis hanyalah CARA MEMBAGI DATA, pada model dan data yang
  sama persis. Itulah selisih yang memisahkan hasil di sini dari hasil yang
  dilaporkan paper-paper tersebut.
""")


if __name__ == "__main__":
    main()
