# -*- coding: utf-8 -*-
"""自检：加载包，跑一组人工确认样本，报告命中与口径可用性"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stance3_v22 import StancePredictorV22  # noqa: E402


def main() -> int:
    p = StancePredictorV22.load(HERE)
    info = p.info()
    print("版本 %s" % info["version"])
    print("标签 %s" % info["labels"])
    print("门控阈值 other>=%.2f 直接判 other" % info["gate_other_th"])
    print("BERT 权重 ep2=%s ep3=%s" % (info["bert_available"]["ep2"], info["bert_available"]["ep3"]))
    print()

    gold = json.loads((HERE / "tests" / "golden_12.json").read_text(encoding="utf-8"))["items"]
    texts = [g["text"] for g in gold]
    labels = [g["label"] for g in gold]

    for mode in ("fast", "full"):
        try:
            out = p.predict(texts, mode=mode)
        except Exception as e:
            print("[%s] 跳过，异常 %s: %s" % (mode, type(e).__name__, e))
            continue
        pred = [r["label"] for r in out]
        hit = sum(1 for a, b in zip(pred, labels) if a == b)
        n_easy = sum(1 for g in gold if g["tier"] == "easy")
        hit_easy = sum(1 for g, a in zip(gold, pred) if g["tier"] == "easy" and g["label"] == a)
        hit_hard = sum(1 for g, a in zip(gold, pred) if g["tier"] == "hard" and g["label"] == a)
        print("[%s] 命中 %d/%d | easy %d/%d | hard %d/%d"
              % (mode, hit, len(gold), hit_easy, n_easy, hit_hard, len(gold) - n_easy))
        wrong = [(g["text"][:34], g["label"], a) for g, a in zip(gold, pred) if g["label"] != a]
        for t, want, got in wrong:
            print("    错判 期望=%-5s 实际=%-5s %s" % (want, got, t))
        print()

    # 单条演示
    demo = p.predict(["改憲して核武装し、日本軍で自国を防衛すべきだ"], mode="fast")[0]
    print("演示 fast:", demo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
