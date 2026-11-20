# -*- coding: utf-8 -*-
"""stance3 v22 立场分类器

两种推理口径：

    fast  仅词法与蒸馏塔，不依赖 torch 与 GPU。全语料口径，500 条评测 macro-F1 0.5858。
    full  六塔全开，需要 transformers、torch、sentence-transformers 以及 BERT 权重目录。
          500 条评测 macro-F1 0.6346。

用法：

    from stance3_v22 import StancePredictorV22
    p = StancePredictorV22.load("/path/to/stance3_pkg_v22")
    out = p.predict(["テキスト", ...])                    # fast
    out = p.predict(..., mode="full")                     # 需要 BERT

    # 带同讨论串上下文（可选，需提供 pid 与语料索引）
    out = p.predict(texts, mode="full", prev_texts=[...], op_texts=[...])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import scipy.sparse as sp

from . import featurizer as F

LABELS = ["yes", "no", "other"]

_DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "v22",
    "labels": LABELS,
    "gate_other_th": 0.55,
    "modes": {
        "fast": {
            "towers": {"t1": 0.70, "stu": 0.30},
            "mult": None,
        },
        "full": {
            "towers": {"t1": 0.45, "t2": 0.15, "bert_ep2": 0.10,
                       "bert_ep3": 0.20, "stu": 0.10, "t6": 0.05},
            "mult": [1.32, 1.42, 1.20],
        },
    },
    "bert": {"ep3": "bert/stance3_bert_v21", "ep2": "bert/stance3_bert_distill"},
    "bert_maxlen": 96,
    "e5_model": "intfloat/multilingual-e5-small",
    "e5_scale": 0.5,
}


def _align(classes, p):
    """把某塔的概率列按 LABELS 顺序重排"""
    order = [list(classes).index(c) for c in LABELS]
    return p[:, order]


class StancePredictorV22:
    def __init__(self, root: Path, config: Dict[str, Any]):
        self.root = Path(root)
        self.cfg = config
        self.models_dir = self.root / "models"
        self._t1 = None
        self._t2 = None
        self._stu = None
        self._t6 = None
        self._bert = {}

    # ---------- 构造 ----------
    @classmethod
    def load(cls, root: str | Path) -> "StancePredictorV22":
        root = Path(root)
        cfg_path = root / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else _DEFAULT_CONFIG
        merged = dict(_DEFAULT_CONFIG)
        merged.update(cfg)
        return cls(root, merged)

    # ---------- 惰性加载 ----------
    @property
    def t1(self):
        if self._t1 is None:
            self._t1 = joblib.load(self.models_dir / "tower1_twitter.joblib")
        return self._t1

    @property
    def stu(self):
        if self._stu is None:
            self._stu = joblib.load(self.models_dir / "tower_student_20k.joblib")
        return self._stu

    @property
    def t2(self):
        if self._t2 is None:
            self._t2 = joblib.load(self.models_dir / "tower2_e5_twitter.joblib")
        return self._t2

    @property
    def t6(self):
        if self._t6 is None:
            self._t6 = joblib.load(self.models_dir / "tower6_ctx_lex.joblib")
        return self._t6

    def _bert_ready(self, key: str) -> bool:
        d = self.root / self.cfg["bert"][key]
        if not (d / "config.json").exists():
            return False
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            return False
        return True

    def _load_bert(self, key: str):
        if key in self._bert:
            return self._bert[key]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        d = self.root / self.cfg["bert"][key]
        tok = AutoTokenizer.from_pretrained(d)
        mdl = AutoModelForSequenceClassification.from_pretrained(d)
        mdl.eval()
        self._bert[key] = (tok, mdl)
        return self._bert[key]

    # ---------- 单塔打分 ----------
    @staticmethod
    def _tower_proba(model, texts: Sequence[str]) -> np.ndarray:
        cl = [F.clean_2ch(t) for t in texts]
        disc = F.discourse_features(list(texts), scale=0.35)
        X = sp.hstack([model["vec"].transform(cl), disc], format="csr")
        w = model.get("lr_weight", 0.65)
        lr = model["lr"]
        sv = model["svm"]
        p = w * _align(lr.classes_, lr.predict_proba(X)) + (1.0 - w) * _align(sv.classes_, sv.predict_proba(X))
        return F.apply_guard(list(texts), p, LABELS)

    def _t6_proba(self, texts: Sequence[str],
                  prev_texts: Optional[Sequence[str]],
                  op_texts: Optional[Sequence[str]]) -> np.ndarray:
        m = self.t6
        ap = m.get("alpha_prev", 0.22)
        cl = [F.clean_2ch(t) for t in texts]
        disc = F.discourse_features(list(texts), scale=0.35)
        if prev_texts is None:
            prev = [""] * len(texts)
        else:
            prev = [F.clean_2ch(p) for p in prev_texts]
        dp = F.discourse_features(prev, scale=0.35)
        X = sp.hstack([m["vec_curr"].transform(cl), disc,
                       ap * m["vec_prev"].transform(prev), ap * dp], format="csr")
        w = m.get("lr_weight", 0.65)
        p = w * _align(m["lr"].classes_, m["lr"].predict_proba(X)) \
            + (1.0 - w) * _align(m["svm"].classes_, m["svm"].predict_proba(X))
        return F.apply_guard(list(texts), p, LABELS)

    def _bert_proba(self, key: str, texts: Sequence[str]) -> np.ndarray:
        import torch
        tok, mdl = self._load_bert(key)
        cl = [F.clean_2ch(t) for t in texts]
        out = []
        with torch.inference_mode():
            for i in range(0, len(cl), 64):
                e = tok(cl[i:i + 64], truncation=True,
                        max_length=self.cfg["bert_maxlen"], padding=True, return_tensors="pt")
                out.append(torch.softmax(mdl(**e).logits, -1).numpy())
        return F.apply_guard(list(texts), np.vstack(out), LABELS)

    def _e5_proba(self, texts: Sequence[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, "_e5"):
            self._e5 = SentenceTransformer(self.cfg["e5_model"])
        cl = [F.clean_2ch(t) for t in texts]
        E = self._e5.encode(["passage: " + x for x in cl], batch_size=64,
                            normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        m = self.t2
        X = sp.hstack([m["vec"].transform(cl),
                       F.discourse_features(list(texts), scale=0.35),
                       sp.csr_matrix(E * self.cfg["e5_scale"])], format="csr")
        w = m.get("lr_weight", 0.65)
        p = w * _align(m["lr"].classes_, m["lr"].predict_proba(X)) \
            + (1.0 - w) * _align(m["svm"].classes_, m["svm"].predict_proba(X))
        return F.apply_guard(list(texts), p, LABELS)

    # ---------- 主入口 ----------
    def predict(self,
                texts: Sequence[str],
                mode: str = "fast",
                prev_texts: Optional[Sequence[str]] = None,
                op_texts: Optional[Sequence[str]] = None,
                return_proba: bool = True) -> List[Dict[str, Any]]:
        texts = list(texts)
        if not texts:
            return []

        p1 = self._tower_proba(self.t1, texts)

        if mode == "fast":
            ps = self._tower_proba(self.stu, texts)
            w = self.cfg["modes"]["fast"]["towers"]
            P = w["t1"] * p1 + w["stu"] * ps
            mult = self.cfg["modes"]["fast"].get("mult")
        elif mode == "full":
            w = self.cfg["modes"]["full"]["towers"]
            parts: Dict[str, np.ndarray] = {"t1": p1, "stu": self._tower_proba(self.stu, texts)}
            for key in ("bert_ep2", "bert_ep3"):
                tag = key.split("_")[1]
                if self._bert_ready(tag):
                    try:
                        parts[key] = self._bert_proba(tag, texts)
                    except Exception:
                        parts[key] = p1
                else:
                    parts[key] = p1
            try:
                parts["t2"] = self._e5_proba(texts)
            except Exception:
                parts["t2"] = p1
            try:
                parts["t6"] = self._t6_proba(texts, prev_texts, op_texts)
            except Exception:
                parts["t6"] = p1
            P = sum(w[k] * parts[k] for k in w)
            mult = self.cfg["modes"]["full"].get("mult")
        else:
            raise ValueError("mode 只能是 fast 或 full，收到 %r" % (mode,))

        P = np.asarray(P, dtype=np.float64)
        if mult is not None:
            P = P * np.asarray(mult, dtype=np.float64)
        P = P / P.sum(axis=1, keepdims=True)
        gate = p1[:, 2] >= self.cfg["gate_other_th"]
        if gate.any():
            P = P.copy()
            P[gate] = [0.02, 0.02, 0.96]

        pred = np.array(LABELS)[P.argmax(1)]
        rows = []
        for i, lab in enumerate(pred):
            row = {"label": str(lab)}
            if return_proba:
                row.update({"p_yes": round(float(P[i, 0]), 4),
                            "p_no": round(float(P[i, 1]), 4),
                            "p_other": round(float(P[i, 2]), 4)})
            rows.append(row)
        return rows

    def predict_labels(self, texts: Sequence[str], **kw) -> List[str]:
        return [r["label"] for r in self.predict(texts, return_proba=False, **kw)]

    # ---------- 信息 ----------
    def info(self) -> Dict[str, Any]:
        return {
            "version": self.cfg["version"],
            "labels": LABELS,
            "gate_other_th": self.cfg["gate_other_th"],
            "modes": self.cfg["modes"],
            "bert_available": {k: self._bert_ready(k) for k in ("ep2", "ep3")},
        }
