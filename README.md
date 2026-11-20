# stance3 v22 立场分类器

日本 2ch 论坛帖子的三分类立场识别包。

三类标签：

| 标签 | 含义 |
|---|---|
| yes | 支持改宪与国防强化议题，含自卫队明记、九条改正、紧急事态条项、防卫费、核武装、间谍防止法 |
| no | 反对上述议题，或支持政府所反对的议题，含移民扩大、夫妇别姓、同性婚 |
| other | 无立场，含转述、纯分析、情绪辱骂、杂谈、碎片 |

## 成绩

两个评估集性质不同，不能合并看。

audit200 为从 589,678 条语料随机抽取的 200 条，代表真实分布，是主指标。
user300 为专挑旧模型漏判样本构造的困难集，不代表分布。

| 口径 | audit200 macro-F1 | audit200 准确率 | user300 macro-F1 | COMB500 macro-F1 |
|---|---:|---:|---:|---:|
| fast | 0.6161 | 69.5% | 0.5691 | 0.5857 |
| full | 0.6769 | 74.0% | 0.6072 | 0.6346 |

audit200 上全判 other 的基线准确率为 59.5%，full 口径净增益 14.5 个百分点。

分项成绩（full 口径，audit200）：

| 类别 | 条数 | 召回率 | 精确率 |
|---|---:|---:|---:|
| yes | 47 | 53.2% | 59.5% |
| no | 34 | 61.8% | 65.6% |
| other | 119 | 85.7% | 81.0% |

已知短板：不含口号词的隐喻与反讽帖会被漏判成 other，占全部错误的约 46%。

## 安装

分三步。第一步解压主包。

```
unzip stance3_pkg_v22_repro.zip -d stance3_pkg_v22
cd stance3_pkg_v22
```

Windows 用资源管理器右键解压到 `stance3_pkg_v22` 目录即可，效果一样。

第二步装依赖。fast 口径只需要 numpy、scipy、scikit-learn、joblib、pandas。

```
pip install -r requirements.txt
```

要用 full 口径再装深度学习依赖。

```
pip install -r requirements-full.txt
```

第三步，可选，解压双 BERT 权重包与全语料打分包到同一目录。

```
unzip stance3_pkg_v22_bert.zip -d stance3_pkg_v22
unzip stance3_pkg_v22_corpus.zip -d stance3_pkg_v22
```

解压后 `bert/` 下会有 `stance3_bert_v21/` 与 `stance3_bert_distill/` 两个目录，`data/` 下会有 `corpus_scores_v22.csv`。

装完跑一次自检与复现校验，确认无误。

```
python selftest.py
python eval/eval_all.py --mode frozen
```

装齐双 BERT 包时，`selftest.py` 会打印 `BERT 权重 ep2=True ep3=True`，`[full] 命中 12/15`。

## 用法

最基本的调用。

```python
import sys
sys.path.insert(0, "/path/to/stance3_pkg_v22")   # 换成实际解压路径
from stance3_v22 import StancePredictorV22

p = StancePredictorV22.load("/path/to/stance3_pkg_v22")

texts = [
    "まるで植民地のような憲法だったから改正は当然。",
    "軍国主義国家に逆戻りさせるつもりかい？",
    "おはようございます",
]

# fast 口径，无需 BERT，CPU 上一秒数千条
for r in p.predict(texts):
    print(r)

# full 口径，需要 BERT 权重
for r in p.predict(texts, mode="full"):
    print(r)
```

返回值为字典列表，每项形如：

```python
{"label": "yes", "p_yes": 0.5669, "p_no": 0.3118, "p_other": 0.1213}
```

只要标签不要概率，用下面这个，速度快一点。

```python
labels = p.predict_labels(texts, mode="fast")
# ["yes", "no", "other"]
```

查看当前包的能力与权重配置。

```python
print(p.info())
# {'version': 'v22', 'labels': [...], 'gate_other_th': 0.55,
#  'modes': {...}, 'bert_available': {'ep2': False, 'ep3': True}}
```

## 批量处理

语料量大时不要一次全部传入，按块处理并分块落盘。

```python
import pandas as pd
from stance3_v22 import StancePredictorV22

p = StancePredictorV22.load("/path/to/stance3_pkg_v22")
df = pd.read_csv("input.csv")          # 需要有一列 text
CHUNK = 40000

rows = []
for i in range(0, len(df), CHUNK):
    part = df.text.iloc[i:i + CHUNK].astype(str).tolist()
    out = p.predict(part, mode="full")
    for t, r in zip(part, out):
        rows.append({"text": t, **r})
    print("%d / %d" % (i + len(part), len(df)))

pd.DataFrame(rows).to_csv("output.csv", index=False, encoding="utf-8-sig")
```

速度参考。本机 18 线程 CPU，fast 口径全量 589,678 条约 5 分钟，full 口径因为要走 BERT，约 36 条每秒，全量需 7.5 小时。有 GPU 时 full 会快很多。

## 输出字段的下游用法

三个概率之和为 1。label 是三者取最大值的结果，但有一条例外：当 t1 塔输出的 other 概率不低于 0.55 时，直接判 other，见下面的门控规则。

统计三类占比。

```python
import collections
cnt = collections.Counter(r["label"] for r in out)
total = sum(cnt.values())
for k in ("other", "yes", "no"):
    print("%-5s %.2f%%" % (k, cnt.get(k, 0) / total * 100))
```

需要调整立场判定松紧时，不要改模型，改阈值即可。下面的写法相当于把判定为有立场的门槛放宽。

```python
import numpy as np
L3 = ["yes", "no", "other"]
for r in out:
    pr = np.array([r["p_yes"], r["p_no"], r["p_other"]])
    if pr[2] < 0.45:                       # 默认是取最大值，这里换成按阈值
        r["label"] = "yes" if pr[0] >= pr[1] else "no"
```

注意改阈值会同时影响三类占比，与 README 末尾的已知限制一起看。

## 带讨论串上下文的调用

full 口径的 t6 塔会使用同讨论串前文。若手上没有前文，传 None 即可，t6 会退化为只用本帖特征。

若手上有带 pid 的语料表，用下面的方式构造前文：

```python
from stance3_v22 import build_context

prev_texts, op_texts = build_context(pid_list, text_list)
out = p.predict(text_list, mode="full", prev_texts=prev_texts, op_texts=op_texts)
```

pid 格式为 `2ch/<board> <thread>.seg#<楼层号>`，其中 # 前为讨论串标识，同一串的帖子需要连续排列。

## BERT 权重

`bert/` 目录默认是空的，因为这个包只带词法塔与蒸馏塔。

要让 full 口径完整生效，把两个权重目录放进去：

```
bert/
  stance3_bert_v21/      # cl-tohoku/bert-base-japanese-v3 微调，第 3 轮
  stance3_bert_distill/  # 同一训练，第 2 轮
```

缺任意一个时，full 口径会用 t1 的概率替代该塔，并在 info() 里报告缺失。

## 门控规则

当 t1 输出的 other 概率不低于 0.55 时，直接判定为 other，不再走融合。

该规则来自实测：全语料中三成帖子属于无立场闲聊，跳过这部分能减少推理量，同时抑制 BERT 在闲聊帖上的假阳性。

## 已知限制

一，全量口径的 other 占比为 71.06%，而随机样本的真实占比是 59.5%，低估了立场帖约 11.5 个百分点。原因是立场召回率只有 53% 到 62%。

二，全量表 no 占比 11.07%，参考值 12.58%，低估 1.5 个百分点。若用于少数派声量统计，建议改用 full 口径并对 no 做阈值调整。

三，训练用的人工 2ch 标注只有 886 条，占语料 0.15%。蒸馏塔的 20000 条标签为模型预测而非人工，误差会被固化。

四，本包在 CPU 上可用，full 口径的 BERT 推理速度约 36 条每秒。

## 自检

```
python selftest.py
```

会跑 12 条人工确认过的样本，打印命中数与两种口径的可用性。

## 文件说明

```
stance3_pkg_v22/
  config.json              权重、校准系数、门控阈值、评测指标
  requirements.txt         fast 口径依赖，已锁版本
  requirements-full.txt    full 口径额外依赖
  README.md                本文件，面向使用
  AGENT_SETUP.md           面向自动化代理的配置契约
  REPRODUCE.md             复现指南，每个数字怎么复现
  selftest.py              冒烟测试

  stance3_v22/             推理代码
    __init__.py
    predictor.py           主入口
    context.py             同讨论串前文提取
    featurizer.py          清洗与 35 维语篇特征

  models/                  四个塔的 joblib
    tower1_twitter.joblib      t1 词法塔
    tower2_e5_twitter.joblib   t2 E5 混合塔
    tower_student_20k.joblib   stu 蒸馏塔
    tower6_ctx_lex.joblib      t6 前文塔

  bert/
    stance3_bert_v21/      ep3 权重，融合权重 0.20
    stance3_bert_distill/  ep2 权重，融合权重 0.10

  data/
    eval_audit200.csv            200 条随机评估集，主指标来源
    eval_user300.csv             300 条对抗评估集
    labels_2ch_train_886.csv     886 条 2ch 人工标注
    train_pool_v4_2721.csv       训练池
    corpus_scores_v22.csv        589678 条全语料打分结果
    frozen_probs/                九个冻结概率，供无 BERT 复现

  train/                   训练脚本
    train_towers_v22.py
    train_bert_v21.py

  eval/                    复现脚本
    eval_all.py
    eval_funnel_audit200.py
    expected_metrics.json

  docs/
    RUBRIC_v3.md                 标注手册，含 H1 至 H5 五条语篇规则
    technical_report_v22.md      技术报告
    experiment_log_v21_v22.md    实验记录
    hpc_plan.md                  多卡 GPU 方案

  examples/predict_demo.py
  tests/golden_12.json
```

## 复现

```
pip install -r requirements.txt
python eval/eval_all.py --mode frozen     # 10 秒，不需要 BERT
```

期望 12 项全部通过。完整路径与失败排查见 `REPRODUCE.md`。
