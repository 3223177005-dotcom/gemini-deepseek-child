# -*- coding: utf-8 -*-
"""自包含 BERT 微调脚本：使用包内 data/train_pool_v4_2721.csv 微调 cl-tohoku/bert-base-japanese-v3
所有路径均相对于包根目录 ROOT，无需外部绝对路径。
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from stance3_v22 import featurizer as F  # noqa: E402

D = ROOT / "data"
L3 = ["yes", "no", "other"]
L2I = {c: i for i, c in enumerate(L3)}
CW = {"yes": 1.0, "no": 1.45, "other": 1.30}
MODEL = "cl-tohoku/bert-base-japanese-v3"
MAXLEN, BS, EPOCHS, LR = 96, 16, 3, 2e-5
OUT = ROOT / "bert" / "stance3_bert_v21"

torch.manual_seed(42)
np.random.seed(42)

pool = pd.read_csv(D / "train_pool_v4_2721.csv", keep_default_na=False)
aud = pd.read_csv(D / "eval_audit200.csv", keep_default_na=False)
u3 = pd.read_csv(D / "eval_user300.csv", keep_default_na=False)
print("[DATA] train %d | aud %d | u3 %d" % (len(pool), len(aud), len(u3)), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
enc = model.bert if hasattr(model, "bert") else model.base_model
for i, layer in enumerate(enc.encoder.layer):
    if i < 6:
        for p in layer.parameters():
            p.requires_grad = False
print("[MODEL] frozen bottom 6/12", flush=True)


class DS(Dataset):
    def __init__(self, texts, labels=None):
        self.t = [F.clean_2ch(x) for x in texts]
        self.y = labels

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = tok(self.t[i], truncation=True, max_length=MAXLEN, padding="max_length")
        it = {k: torch.tensor(v) for k, v in e.items()}
        if self.y is not None:
            it["labels"] = torch.tensor(L2I[self.y[i]])
        return it


dl_tr = DataLoader(DS(pool.text.tolist(), pool.label.tolist()), batch_size=BS, shuffle=True)
dl_a = DataLoader(DS(aud.text.tolist()), batch_size=32)
dl_u = DataLoader(DS(u3.text.tolist()), batch_size=32)

lossf = nn.CrossEntropyLoss(weight=torch.tensor([CW[c] for c in L3], dtype=torch.float32))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)


def infer(dl):
    model.eval()
    out = []
    with torch.no_grad():
        for b in dl:
            b.pop("labels", None)
            out.append(torch.softmax(model(**b).logits, -1).numpy())
    return np.vstack(out)


def report(tag):
    Pa = F.apply_guard(aud.text.tolist(), infer(dl_a), L3)
    Pu = F.apply_guard(u3.text.tolist(), infer(dl_u), L3)
    pa = np.array(L3)[Pa.argmax(1)]
    pu = np.array(L3)[Pu.argmax(1)]
    fa = f1_score(aud.label_v3, pa, average=None, labels=L3, zero_division=0)
    fu = f1_score(u3.user_label, pu, average=None, labels=L3, zero_division=0)
    ma = f1_score(aud.label_v3, pa, average="macro", labels=L3, zero_division=0)
    mu = f1_score(u3.user_label, pu, average="macro", labels=L3, zero_division=0)
    print("%-6s AUD %.4f (y%.3f n%.3f o%.3f) | U300 %.4f (y%.3f n%.3f o%.3f) | avg %.4f"
          % (tag, ma, fa[0], fa[1], fa[2], mu, fu[0], fu[1], fu[2], (ma + mu) / 2), flush=True)
    return ma, mu, (ma + mu) / 2


best = None
for ep in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    tot = 0.0
    for i, b in enumerate(dl_tr):
        opt.zero_grad()
        loss = lossf(model(**b).logits, b["labels"])
        loss.backward()
        opt.step()
        tot += float(loss.detach())
        if (i + 1) % 40 == 0:
            print("  ep%d step %d/%d loss %.4f (%.0fs)"
                  % (ep, i + 1, len(dl_tr), tot / (i + 1), time.time() - t0), flush=True)
    print("[EP%d] loss %.4f | %.0fs" % (ep, tot / len(dl_tr), time.time() - t0), flush=True)
    ma, mu, avg = report("ep%d" % ep)
    if best is None or avg > best[0]:
        best = (avg, ep, ma, mu)
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)

print("\n[BEST] ep%d | AUD %.4f | U300 %.4f | avg %.4f" % (best[1], best[2], best[3], best[0]))
print("DONE-BERT-V21")
