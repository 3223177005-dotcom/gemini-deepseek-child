# -*- coding: utf-8 -*-
"""复现脚本

用法：
    python eval/eval_all.py --mode frozen      # 不用 BERT，验证融合数学与词法塔，秒级
    python eval/eval_all.py --mode live        # 真跑全部塔，需要 BERT 与 E5，full 约数分钟
    python eval/eval_all.py --mode all         # 两者都跑，并核对全语料分布
    python eval/eval_all.py --mode all --corpus  # 额外校验 589,678 条打分表

判据来自 eval/expected_metrics.json，逐项比对，容差 0.0005。

frozen 模式的做法：t1 与 stu 两个塔现场计算（快，不依赖 BERT），
t2、bert_ep2、bert_ep3、t6 四个塔读取 data/frozen_probs 下的冻结概率。
只要融合系数与门控写对，就必须复现出 0.6161 与 0.6769。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stance3_v22 import LABELS, StancePredictorV22, build_context  # noqa: E402
from stance3_v22 import featurizer as F  # noqa: E402

L3 = LABELS
FP = ROOT / "data" / "frozen_probs"
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
EXP = json.loads((ROOT / "eval" / "expected_metrics.json").read_text(encoding="utf-8"))
TOL = EXP.get("tolerance", 0.0005)

RESULTS = []


def check(name: str, got: float, want: float, tol: float = TOL) -> None:
    ok = abs(float(got) - float(want)) <= tol
    RESULTS.append((name, ok, float(got), float(want)))
    print("  %-42s 实际 %.4f  期望 %.4f  %s"
          % (name, got, want, "PASS" if ok else "FAIL"))


def load_eval():
    a = pd.read_csv(ROOT / "data" / "eval_audit200.csv", keep_default_na=False)
    u = pd.read_csv(ROOT / "data" / "eval_user300.csv", keep_default_na=False)
    return a, u


def metrics(ya, yu, pa, pu):
    yc = np.concatenate([ya, yu])
    pc = np.concatenate([pa, pu])
    out = {
        "aud_f1": f1_score(ya, pa, average="macro", labels=L3),
        "aud_acc": accuracy_score(ya, pa),
        "u3_f1": f1_score(yu, pu, average="macro", labels=L3),
        "u3_acc": accuracy_score(yu, pu),
        "comb_f1": f1_score(yc, pc, average="macro", labels=L3),
        "comb_acc": accuracy_score(yc, pc),
        "comb_no_f1": f1_score(yc, pc, average=None, labels=L3)[1],
        "aud_per": f1_score(ya, pa, average=None, labels=L3),
        "cm": confusion_matrix(ya, pa, labels=L3),
    }
    return out


def fuse(parts, weights, mult, p1, gate_th):
    P = sum(weights[k] * parts[k] for k in weights)
    P = np.asarray(P, dtype=np.float64)
    if mult is not None:
        P = P * np.asarray(mult, dtype=np.float64)
    P = P / P.sum(axis=1, keepdims=True)
    g = p1[:, 2] >= gate_th
    if g.any():
        P[g] = [0.02, 0.02, 0.96]
    return np.array(L3)[P.argmax(1)]


def tower_probs(model, texts):
    cl = [F.clean_2ch(t) for t in texts]
    disc = F.discourse_features(list(texts), scale=0.35)
    X = sp.hstack([model["vec"].transform(cl), disc], format="csr")
    w = model.get("lr_weight", 0.65)
    o1 = [list(model["lr"].classes_).index(c) for c in L3]
    o2 = [list(model["svm"].classes_).index(c) for c in L3]
    P = w * model["lr"].predict_proba(X)[:, o1] + (1 - w) * model["svm"].predict_proba(X)[:, o2]
    return F.apply_guard(list(texts), P, L3)


def run_frozen():
    print("\n[frozen] 用冻结概率验证融合数学，不依赖 BERT")
    if not FP.exists() or not any(FP.iterdir()):
        print("  跳过：data/frozen_probs 为空")
        return
    aud, u3 = load_eval()
    ta = aud.text.astype(str).tolist()
    tu = u3.text.astype(str).tolist()
    ya = aud[EXP["evaluation_sets"]["audit200"]["label_column"]].values
    yu = u3[EXP["evaluation_sets"]["user300"]["label_column"]].values

    t1 = joblib.load(ROOT / "models" / "tower1_twitter.joblib")
    stu = joblib.load(ROOT / "models" / "tower_student_20k.joblib")

    def stack(name):
        return np.vstack([np.load(FP / ("%s_aud_proba.npy" % name)),
                          np.load(FP / ("%s_u3_proba.npy" % name))])

    P1 = np.vstack([tower_probs(t1, ta), tower_probs(t1, tu)])
    Pst = np.vstack([tower_probs(stu, ta), tower_probs(stu, tu)])

    # 与冻结的 t1 概率比对，用于发现 featurizer 漂移
    f1_frozen = stack("T1tw")
    print("  t1 现场计算与冻结值最大绝对偏差 %.6f" % float(np.abs(P1 - f1_frozen).max()))

    fw = CFG["modes"]["fast"]["towers"]
    fm = CFG["modes"]["fast"].get("mult")
    pred = fuse({"t1": P1, "stu": Pst}, fw, fm, P1, CFG["gate_other_th"])
    m = metrics(ya, yu, pred[:200], pred[200:])
    check("fast / audit200 macro-F1", m["aud_f1"], EXP["fast"]["macro_f1"]["audit200"])
    check("fast / user300 macro-F1", m["u3_f1"], EXP["fast"]["macro_f1"]["user300"])
    check("fast / comb500 macro-F1", m["comb_f1"], EXP["fast"]["macro_f1"]["comb500"])
    check("fast / comb500 no F1", m["comb_no_f1"], EXP["fast"]["comb500_no_f1"])

    parts = {"t1": P1, "stu": Pst, "t2": stack("T2best"),
             "bert_ep2": np.load(FP / "T3ep2_500_proba.npy"),
             "bert_ep3": np.load(FP / "T3_aud_proba.npy") if False else
             np.vstack([np.load(FP / "T3_aud_proba.npy"), np.load(FP / "T3_u3_proba.npy")]),
             "t6": np.load(FP / "T6_500_proba.npy")}
    fw2 = CFG["modes"]["full"]["towers"]
    fm2 = CFG["modes"]["full"].get("mult")
    pred2 = fuse(parts, fw2, fm2, P1, CFG["gate_other_th"])
    m2 = metrics(ya, yu, pred2[:200], pred2[200:])
    check("full / audit200 macro-F1", m2["aud_f1"], EXP["full"]["macro_f1"]["audit200"])
    check("full / user300 macro-F1", m2["u3_f1"], EXP["full"]["macro_f1"]["user300"])
    check("full / comb500 macro-F1", m2["comb_f1"], EXP["full"]["macro_f1"]["comb500"])
    check("full / comb500 no F1", m2["comb_no_f1"], EXP["full"]["comb500_no_f1"])
    check("full / audit200 准确率", m2["aud_acc"], EXP["full"]["accuracy"]["audit200"])
    check("full / audit200 yes F1", m2["aud_per"][0], EXP["full"]["audit200_per_class_f1"]["yes"])
    check("full / audit200 no F1", m2["aud_per"][1], EXP["full"]["audit200_per_class_f1"]["no"])
    check("full / audit200 other F1", m2["aud_per"][2], EXP["full"]["audit200_per_class_f1"]["other"])

    print("\n  full 口径 audit200 混淆矩阵（行=真实 [yes,no,other]）")
    for row in m2["cm"]:
        print("   ", " ".join("%4d" % v for v in row))


def run_live():
    print("\n[live] 实际加载全部塔推理")
    p = StancePredictorV22.load(ROOT)
    info = p.info()
    print("  BERT 可用性 ep2=%s ep3=%s" % (info["bert_available"]["ep2"], info["bert_available"]["ep3"]))
    aud, u3 = load_eval()
    ta = aud.text.astype(str).tolist()
    tu = u3.text.astype(str).tolist()
    ya = aud[EXP["evaluation_sets"]["audit200"]["label_column"]].values
    yu = u3[EXP["evaluation_sets"]["user300"]["label_column"]].values

    pa_prev = aud["prev_text"].astype(str).tolist() if "prev_text" in aud.columns else None
    pa_op = aud["op_text"].astype(str).tolist() if "op_text" in aud.columns else None
    pu_prev = u3["prev_text"].astype(str).tolist() if "prev_text" in u3.columns else None
    pu_op = u3["op_text"].astype(str).tolist() if "op_text" in u3.columns else None

    for mode in ("fast", "full"):
        pa = np.array([r["label"] for r in p.predict(ta, mode=mode, prev_texts=pa_prev, op_texts=pa_op)])
        pu = np.array([r["label"] for r in p.predict(tu, mode=mode, prev_texts=pu_prev, op_texts=pu_op)])
        m = metrics(ya, yu, pa, pu)
        check("%s / audit200 macro-F1" % mode, m["aud_f1"], EXP[mode]["macro_f1"]["audit200"])
        check("%s / user300 macro-F1" % mode, m["u3_f1"], EXP[mode]["macro_f1"]["user300"])
        check("%s / comb500 macro-F1" % mode, m["comb_f1"], EXP[mode]["macro_f1"]["comb500"])

    if not (info["bert_available"]["ep2"] and info["bert_available"]["ep3"]):
        print("  提示：BERT 未装全，live 的 full 数值会低于期望，属预期。"
              "完整复现需 bert/stance3_bert_v21 与 bert/stance3_bert_distill 同时存在。")


def run_corpus():
    print("\n[corpus] 校验 589,678 条打分表")
    fp = ROOT / "data" / "corpus_scores_v22.csv"
    if not fp.exists():
        print("  跳过：data/corpus_scores_v22.csv 不存在")
        return
    df = pd.read_csv(fp, usecols=["pred", "pred_v21_1", "pred_v21_2"])
    n = len(df)
    print("  行数 %d" % n)
    check("corpus 行数", n, EXP["corpus"]["rows"], tol=0.5)
    for col, key in (("pred", "distribution_v22"),
                     ("pred_v21_1", "distribution_v21_1"),
                     ("pred_v21_2", "distribution_v21_2")):
        vc = df[col].value_counts(normalize=True)
        for c in L3:
            check("corpus %s / %s 占比" % (key, c), vc.get(c, 0), EXP["corpus"][key][c])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen", "live", "all"], default="all")
    ap.add_argument("--corpus", action="store_true", help="额外校验全语料打分表")
    args = ap.parse_args()

    print("包版本 %s | 容差 %.4f" % (CFG["version"], TOL))
    if args.mode in ("frozen", "all"):
        run_frozen()
    if args.mode in ("live", "all"):
        run_live()
    if args.corpus:
        run_corpus()

    bad = [r for r in RESULTS if not r[1]]
    print("\n合计 %d 项，通过 %d 项，失败 %d 项" % (len(RESULTS), len(RESULTS) - len(bad), len(bad)))
    if bad:
        print("失败项：")
        for name, _, got, want in bad:
            print("  %-42s 实际 %.4f 期望 %.4f" % (name, got, want))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
