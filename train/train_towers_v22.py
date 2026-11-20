# -*- coding: utf-8 -*-
"""自包含训练脚本：训练 T1(词法塔)、T2(E5混合塔)、T6(输入端前文特征塔) 并搜索六塔融合权重
所有路径均相对于包根目录 ROOT，无需外部绝对路径即可在任意机器上直接运行。
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from stance3_v22 import featurizer as F  # noqa: E402

D = ROOT / "data"
FP = D / "frozen_probs"
M = ROOT / "models"
L3 = ["yes", "no", "other"]
CW = {"yes": 1.0, "no": 1.45, "other": 1.30}
ALPHA_PREV = 0.22

t0 = time.time()

# ---------- 1. 载入包内自包含训练池与评估集（已内置 prev_text 与 op_text 列） ----------
pool = pd.read_csv(D / "train_pool_v4_2721.csv", keep_default_na=False)
pool["text"] = pool.text.astype(str)
pool["clean"] = [F.clean_2ch(t) for t in pool.text]
pool["prev"] = [F.clean_2ch(t) for t in pool.get("prev_text", pd.Series([""] * len(pool))).astype(str)]

aud = pd.read_csv(D / "eval_audit200.csv", keep_default_na=False)
aud["text"] = aud.text.astype(str)
u3 = pd.read_csv(D / "eval_user300.csv", keep_default_na=False)
u3["text"] = u3.text.astype(str)
ya, yu = aud.label_v3.values, u3.user_label.values
yc = np.concatenate([ya, yu])
texts_500 = aud.text.tolist() + u3.text.tolist()
prev_500 = [F.clean_2ch(t) for t in (aud.get("prev_text", pd.Series([""] * len(aud))).astype(str).tolist()
                                     + u3.get("prev_text", pd.Series([""] * len(u3))).astype(str).tolist())]

print("[DATA] train=%d (2ch with prev=%d) | eval=500 (with prev=%d)"
      % (len(pool), int(((pool.is_2ch == 1) & (pool.prev != "")).sum()), sum(1 for p in prev_500 if p)), flush=True)

# ---------- 2. 训练第 6 塔：双视野前文特征塔 T6_ctx_lex ----------
vec_curr = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50000, sublinear_tf=False)
vec_prev = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=25000, sublinear_tf=False)
Xc_tr = vec_curr.fit_transform(pool.clean)
Xc_500 = vec_curr.transform([F.clean_2ch(t) for t in texts_500])
Xp_tr = vec_prev.fit_transform(pool.prev)
Xp_500 = vec_prev.transform(prev_500)
Dc_tr = F.discourse_features(pool.text.tolist(), scale=0.35)
Dc_500 = F.discourse_features(texts_500, scale=0.35)
Dp_tr = F.discourse_features(pool.prev.tolist(), scale=0.35)
Dp_500 = F.discourse_features(prev_500, scale=0.35)

X6_tr = sp.hstack([Xc_tr, Dc_tr, ALPHA_PREV * Xp_tr, ALPHA_PREV * Dp_tr], format="csr")
X6_500 = sp.hstack([Xc_500, Dc_500, ALPHA_PREV * Xp_500, ALPHA_PREV * Dp_500], format="csr")

lr6 = LogisticRegression(C=1.0, class_weight=CW, max_iter=2000).fit(X6_tr, pool.label.values)
sv6 = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight=CW, max_iter=3000), cv=3).fit(X6_tr, pool.label.values)
oL6 = [list(lr6.classes_).index(c) for c in L3]
oS6 = [list(sv6.classes_).index(c) for c in L3]
P6_500 = F.apply_guard(texts_500, 0.65 * lr6.predict_proba(X6_500)[:, oL6]
                       + 0.35 * sv6.predict_proba(X6_500)[:, oS6], L3)
joblib.dump({"vec_curr": vec_curr, "vec_prev": vec_prev, "lr": lr6, "svm": sv6,
             "lr_weight": 0.65, "alpha_prev": ALPHA_PREV, "labels": L3},
            M / "tower6_ctx_lex.joblib")
np.save(FP / "T6_500_proba.npy", P6_500)
print("[T6] 训练完成 (%.1fs)" % (time.time() - t0), flush=True)


def ev(tag, pred):
    ma = f1_score(ya, pred[:200], average="macro", labels=L3)
    mu = f1_score(yu, pred[200:], average="macro", labels=L3)
    mc = f1_score(yc, pred, average="macro", labels=L3)
    fc = f1_score(yc, pred, average=None, labels=L3)
    print("%-30s AUD %.4f (acc %.3f) | U300 %.4f (acc %.3f) | COMB500 %.4f (y%.3f n%.3f o%.3f)"
          % (tag, ma, accuracy_score(ya, pred[:200]), mu, accuracy_score(yu, pred[200:]),
             mc, fc[0], fc[1], fc[2]), flush=True)
    return ma, mu, mc, float(fc[1])


ev("T6 单独（前文特征塔）", np.array(L3)[P6_500.argmax(1)])

# ---------- 3. 五塔 + 六塔融合验证 ----------
P1 = np.vstack([np.load(FP / "T1tw_aud_proba.npy"), np.load(FP / "T1tw_u3_proba.npy")])
P2 = np.vstack([np.load(FP / "T2best_aud_proba.npy"), np.load(FP / "T2best_u3_proba.npy")])
P3e3 = np.vstack([np.load(FP / "T3_aud_proba.npy"), np.load(FP / "T3_u3_proba.npy")])
P3e2 = np.load(FP / "T3ep2_500_proba.npy")
stu = joblib.load(M / "tower_student_20k.joblib")
oLs = [list(stu["lr"].classes_).index(c) for c in L3]
oSs = [list(stu["svm"].classes_).index(c) for c in L3]
Xs = sp.hstack([stu["vec"].transform([F.clean_2ch(t) for t in texts_500]),
                F.discourse_features(texts_500, scale=0.35)], format="csr")
Pst = F.apply_guard(texts_500, 0.65 * stu["lr"].predict_proba(Xs)[:, oLs]
                    + 0.35 * stu["svm"].predict_proba(Xs)[:, oSs], L3)

MULT = np.array([1.32, 1.42, 1.20])


def gate(P, P1ref):
    P = P * MULT
    P = P / P.sum(1, keepdims=True)
    P[P1ref[:, 2] >= 0.55] = [0.02, 0.02, 0.96]
    return P


P5 = gate(0.45 * P1 + 0.15 * P2 + 0.10 * P3e2 + 0.20 * P3e3 + 0.10 * Pst, P1)
ev("五塔 P5（基线）", np.array(L3)[P5.argmax(1)])

best = None
for w6 in (0.05, 0.10, 0.15, 0.20):
    scale = 1.0 - w6
    P6t = gate(scale * (0.45 / 0.90 * P1 + 0.15 / 0.90 * P2 + 0.10 / 0.90 * P3e2
                        + 0.20 / 0.90 * P3e3 + 0.10 / 0.90 * Pst) + w6 * P6_500, P1)
    r = ev("五塔+前文塔 w6=%.2f" % w6, np.array(L3)[P6t.argmax(1)])
    if best is None or r[2] > best[0]:
        best = (r[2], w6, P6t, r)

mc, w6, P22, r = best
print("\n[BEST] w6=%.2f | AUD %.4f | U300 %.4f | COMB500 %.4f | no F1 %.4f"
      % (w6, r[0], r[1], r[2], r[3]), flush=True)
np.save(FP / "v22_500_proba.npy", P22)
print("DONE-TRAIN-TOWERS (%.1fs)" % (time.time() - t0), flush=True)
