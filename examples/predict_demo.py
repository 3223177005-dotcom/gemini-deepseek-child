# -*- coding: utf-8 -*-
"""最小调用示例"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from stance3_v22 import StancePredictorV22  # noqa: E402

p = StancePredictorV22.load(HERE.parent)

texts = [
    "改憲賛成。9条に自衛隊を明記すべきだ。",
    "憲法9条を守れ。改憲反対。",
    "まるで植民地のような憲法だったから改正は当然。",
    "おはようございます",
]

print("=== fast 口径 ===")
for t, r in zip(texts, p.predict(texts, mode="fast")):
    print("%-5s y=%.2f n=%.2f o=%.2f  %s"
          % (r["label"], r["p_yes"], r["p_no"], r["p_other"], t[:38]))

print()
print("=== full 口径（需 BERT 权重，缺失时自动退化）===")
for t, r in zip(texts, p.predict(texts, mode="full")):
    print("%-5s y=%.2f n=%.2f o=%.2f  %s"
          % (r["label"], r["p_yes"], r["p_no"], r["p_other"], t[:38]))
