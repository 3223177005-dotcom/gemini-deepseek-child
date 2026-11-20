# -*- coding: utf-8 -*-
"""复现 audit200 分歧区带前文的级联漏斗结果

原始流程：找出 audit200 中 t1 与 v22 意见不一致或 margin 低于 0.10 的样本，
导出【首楼 OP】+【上一楼或引用楼 PREV】+【本帖 CURR】三段式卡片，
人工盲判后覆盖，比较覆盖前后的 macro-F1。

本脚本复现该结果。盲判结论为人工产物，已固定在本文件的 LAB 字典里。
使用的概率来自 data/frozen_probs，不依赖 BERT。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stance3_v22 import LABELS  # noqa: E402

L3 = LABELS
FP = ROOT / "data" / "frozen_probs"

# 人工盲判结论，键为 audit200 行号
LAB = {
    3: "yes", 5: "other", 9: "yes", 12: "no", 13: "no", 18: "no", 27: "yes",
    31: "no", 32: "yes", 34: "other", 35: "yes", 41: "yes", 42: "no",
    46: "other", 54: "other", 57: "yes", 58: "other", 59: "other",
    65: "no", 66: "no", 67: "other", 70: "other", 74: "other", 78: "other",
    80: "other", 83: "other", 85: "other",
    86: "other", 87: "yes", 93: "no", 94: "yes", 95: "other", 102: "yes",
    103: "other", 115: "other", 122: "other", 127: "other", 129: "other",
    135: "no", 139: "other", 143: "other", 144: "no", 146: "yes",
    147: "no", 150: "other", 151: "other", 158: "no", 163: "yes",
    164: "other", 169: "no", 174: "other", 185: "no", 193: "other", 195: "yes",
}

EXPECT = {"auto": 0.6769, "funnel": 0.6934, "auto_acc": 0.7400,
          "funnel_acc": 0.7550, "blind_agreement": 0.6300,
          "divergent_auto_acc": 0.5741, "n": 54}
TOL = 0.0005


def main() -> int:
    results = []

    def chk(name, got, want):
        ok = abs(float(got) - float(want)) <= TOL
        results.append(ok)
        print("  %-34s 实际 %.4f  期望 %.4f  %s"
              % (name, got, want, "PASS" if ok else "FAIL"))

    aud = pd.read_csv(ROOT / "data" / "eval_audit200.csv", keep_default_na=False)
    ya = aud["label_v3"].values
    P22 = np.load(FP / "v22_500_proba.npy")[:200]
    pred_auto = np.array(L3)[P22.argmax(1)]

    chk("分歧区样本数", len(LAB), EXPECT["n"])
    chk("纯自动 macro-F1", f1_score(ya, pred_auto, average="macro", labels=L3), EXPECT["auto"])
    chk("纯自动 准确率", accuracy_score(ya, pred_auto), EXPECT["auto_acc"])

    pred_fn = pred_auto.copy()
    for i, lab in LAB.items():
        pred_fn[i] = lab
    chk("漏斗后 macro-F1", f1_score(ya, pred_fn, average="macro", labels=L3), EXPECT["funnel"])
    chk("漏斗后 准确率", accuracy_score(ya, pred_fn), EXPECT["funnel_acc"])

    agree = float(np.mean([LAB[i] == ya[i] for i in LAB]))
    chk("盲判与 label_v3 一致率", agree, EXPECT["blind_agreement"])
    div_acc = float(np.mean([pred_auto[i] == ya[i] for i in LAB]))
    chk("分歧区纯自动正确率", div_acc, EXPECT["divergent_auto_acc"])

    f0 = f1_score(ya, pred_auto, average=None, labels=L3)
    f1v = f1_score(ya, pred_fn, average=None, labels=L3)
    print("\n分项 F1")
    print("  %-6s %-8s %-8s" % ("类别", "纯自动", "漏斗后"))
    for i, c in enumerate(L3):
        print("  %-6s %.4f   %.4f" % (c, f0[i], f1v[i]))

    print("\n漏斗后混淆矩阵（行=真实 [yes,no,other]）")
    for row in confusion_matrix(ya, pred_fn, labels=L3):
        print("   ", " ".join("%4d" % v for v in row))

    bad = results.count(False)
    print("\n合计 %d 项，通过 %d 项，失败 %d 项"
          % (len(results), len(results) - bad, bad))
    if bad:
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
