# 交付清单

完整项目交付分三个压缩包。解压到同一目录后即构成完整工程。

## 一、三个包

| 包名 | 体积 | 内容 | 是否必需 |
|---|---:|---|---|
| `stance3_pkg_v22_repro.zip` | 约 23 MB | 代码、文档、四个模型塔、评估集、标注数据、冻结概率、训练脚本、复现脚本 | 必需 |
| `stance3_pkg_v22_bert.zip` | 约 800 MB | `bert/stance3_bert_v21`（ep3）与 `bert/stance3_bert_distill`（ep2） | 复现 full 口径需要 |
| `stance3_pkg_v22_corpus.zip` | 约 55 MB | `data/corpus_scores_v22.csv`，589,678 条全语料打分 | 需要现成结果时需要 |

## 二、组装方式

三个包都解压到同一个 `stance3_pkg_v22` 目录，会自然合并成下面的结构。

```
stance3_pkg_v22/
  config.json
  requirements.txt
  requirements-full.txt
  README.md
  AGENT_SETUP.md
  REPRODUCE.md
  selftest.py
  stance3_v22/          推理代码
  models/               四个 joblib
  bert/                 两个 BERT 权重目录
  data/                 评估集、标注、冻结概率、全语料打分
  train/                训练脚本
  eval/                 复现脚本
  docs/                 标注手册、技术报告、实验记录、HPC 方案
  examples/
  tests/
```

Linux 与 macOS：

```
mkdir -p stance3_pkg_v22
unzip stance3_pkg_v22_repro.zip -d stance3_pkg_v22
unzip stance3_pkg_v22_bert.zip -d stance3_pkg_v22
unzip stance3_pkg_v22_corpus.zip -d stance3_pkg_v22
```

Windows PowerShell：

```
Expand-Archive stance3_pkg_v22_repro.zip -DestinationPath stance3_pkg_v22
Expand-Archive stance3_pkg_v22_bert.zip -DestinationPath stance3_pkg_v22
Expand-Archive stance3_pkg_v22_corpus.zip -DestinationPath stance3_pkg_v22
```

## 三、验收顺序

解压后按下面三步验证。第一步不通过就不用往后走。

第一步，冒烟测试。

```
cd stance3_pkg_v22
pip install -r requirements.txt
python selftest.py
```

看五行输出是否与 `AGENT_SETUP.md` 第 3 节一致。

第二步，快速复现，不需要 BERT。

```
python eval/eval_all.py --mode frozen
```

应输出 `合计 12 项，通过 12 项，失败 0 项`。这一步验证融合系数、校准系数、门控规则、词法塔与蒸馏塔的可复现性。

第三步，完整复现，需要 BERT 与额外依赖。

```
pip install -r requirements-full.txt
python eval/eval_all.py --mode live
```

应输出 `合计 6 项，通过 6 项，失败 0 项`。这一步从零加载四个模型塔实际推理，不做缓存。

## 四、核心数字

主指标为 `audit200`，它是从 589,678 条语料随机抽取的 200 条，代表真实分布。

| 口径 | audit200 macro-F1 | audit200 准确率 | user300 | comb500 |
|---|---:|---:|---:|---:|
| fast，无 BERT | 0.6161 | 69.5% | 0.5691 | 0.5857 |
| full，含两 BERT | 0.6769 | 74.0% | 0.6072 | 0.6346 |

`audit200` 上全判 other 的基线准确率为 59.5%，full 口径净增益 14.5 个百分点。

`user300` 的 300 条 100% 抽自旧模型漏判样本，不代表语料分布，不可用于对外陈述分布类结论。

## 五、包内不含的内容

| 项目 | 原因 |
|---|---|
| 2ch 原始抓取流程 | 原始数据体积过大，且含第三方站点内容。只提供打分结果 |
| 20,000 条蒸馏样本的三塔预测 | 需要先跑全语料 BERT 推理约 7.5 小时 |
| 分歧区盲判的原文卡片 | 可由训练脚本重新生成，盲判结论已写入评估脚本 |

## 六、环境

实测环境为 Python 3.13.3、numpy 2.5.2、scipy 1.18.0、scikit-learn 1.9.0、joblib 1.5.3、pandas 3.0.5。

full 口径额外需要 torch 2.15.0、transformers 5.23.0、sentence-transformers 6.3.0。

sklearn 低于 1.9.0 时加载 joblib 可能报 InconsistentVersionWarning 或失败，这是本工程最可能遇到的兼容问题。

## 七、速度

18 线程 CPU，无 GPU。

| 口径 | 速度 | 589,678 条全量 |
|---|---|---|
| fast | 3,006 条每秒 | 约 3.3 分钟 |
| full | 约 36 条每秒 | 约 7.5 小时 |
