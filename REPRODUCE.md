# 复现指南

本文件说明如何复现包内报出的每一个数字。所有数字的权威定义在 `eval/expected_metrics.json`，容差 0.0005。

## 1 目录说明

```
stance3_pkg_v22/
  config.json                    融合权重、校准系数、门控阈值
  requirements.txt               fast 口径依赖
  requirements-full.txt          full 口径额外依赖
  README.md                      使用说明
  AGENT_SETUP.md                 面向自动化代理的配置契约
  REPRODUCE.md                   本文件
  selftest.py                    冒烟测试

  stance3_v22/                   推理代码
    predictor.py
    context.py
    featurizer.py

  models/                        四个塔的 joblib

  bert/
    stance3_bert_v21/            ep3 权重，权重 0.20
    stance3_bert_distill/        ep2 权重，权重 0.10

  data/
    eval_audit200.csv            200 条随机评估集，主指标来源
    eval_user300.csv             300 条对抗评估集
    labels_2ch_train_886.csv     886 条 2ch 人工标注
    train_pool_v4_2721.csv       训练池，推特 1835 加 2ch 886
    corpus_scores_v22.csv        589678 条全语料打分结果
    frozen_probs/                九个冻结概率文件，供无 BERT 复现

  train/
    train_towers_v22.py          训练词法塔、蒸馏塔、前文塔
    train_bert_v21.py            微调 BERT

  eval/
    eval_all.py                  复现主指标
    eval_funnel_audit200.py      复现分歧区漏斗结果
    expected_metrics.json        全部期望值

  docs/
    RUBRIC_v3.md                 标注手册，含 H1 至 H5 五条语篇规则
    technical_report_v22.md      技术报告
    experiment_log_v21_v22.md    实验记录，含各版本演进与失效路线
    hpc_plan.md                  上多卡 GPU 的方案

  examples/predict_demo.py
  tests/golden_12.json
```

## 2 两条复现路径

### 2.1 快速路径，不需要 BERT

```
pip install -r requirements.txt
python eval/eval_all.py --mode frozen
```

耗时约 10 秒。做法是：t1 与 stu 两个塔现场计算，t2、bert_ep2、bert_ep3、t6 四个塔读取 `data/frozen_probs` 下的冻结概率，然后按 config 融合。

期望输出：

```
t1 现场计算与冻结值最大绝对偏差 0.000000
fast / audit200 macro-F1    0.6161
fast / user300 macro-F1     0.5691
fast / comb500 macro-F1     0.5857
fast / comb500 no F1        0.5063
full / audit200 macro-F1    0.6769
full / user300 macro-F1     0.6072
full / comb500 macro-F1     0.6346
full / comb500 no F1        0.5269
full / audit200 准确率       0.7400
full / audit200 yes F1      0.5618
full / audit200 no F1       0.6364
full / audit200 other F1    0.8327
合计 12 项，通过 12 项，失败 0 项
```

这条路径验证的是融合系数、校准系数、门控规则、以及词法塔与蒸馏塔的可复现性。它验证不了 BERT 权重本身。

### 2.2 完整路径，需要 BERT

```
pip install -r requirements.txt -r requirements-full.txt
python eval/eval_all.py --mode live
```

耗时约 2 分钟，含 E5 模型首次下载。这条路径从零加载四个模型塔，对 500 条评估样本实际推理，不做任何缓存。

期望输出：

```
BERT 可用性 ep2=True ep3=True
fast / audit200 macro-F1    0.6161
fast / user300 macro-F1     0.5691
fast / comb500 macro-F1     0.5857
full / audit200 macro-F1    0.6769
full / user300 macro-F1     0.6072
full / comb500 macro-F1     0.6346
合计 6 项，通过 6 项，失败 0 项
```

若 `ep2` 或 `ep3` 显示 False，说明 `bert/` 下对应目录缺失，或者 torch 与 transformers 未装。这种情况下 full 的数值会降到下表对应档位，这不是故障。

### 2.3 全语料表校验

```
python eval/eval_all.py --mode all --corpus
```

校验 `data/corpus_scores_v22.csv` 的行数与三类占比，连同 live 合计 34 项。这一步会跑 live，所以需要 BERT。

只想校验表格、不想跑 BERT 时，用下面这条。

```
python -c "import pandas as pd; d=pd.read_csv('data/corpus_scores_v22.csv',usecols=['pred']); print(len(d), d.pred.value_counts(normalize=True).round(4).to_dict())"
```

期望行数 589678，分布 other 0.7106、yes 0.1787、no 0.1107。

### 2.4 分歧区漏斗复现

```
python eval/eval_funnel_audit200.py
```

复现 audit200 分歧区 54 条带前文盲判后覆盖的结果，共 7 项。不依赖 BERT，秒级完成。期望输出：

```
分歧区样本数                54
纯自动 macro-F1          0.6769
纯自动 准确率             0.7400
漏斗后 macro-F1          0.6934
漏斗后 准确率             0.7550
盲判与 label_v3 一致率     0.6296
分歧区纯自动正确率           0.5741
合计 7 项，通过 7 项，失败 0 项
```

该脚本里的盲判结论是人工产物，写死在脚本的 LAB 字典里，不参与任何训练。一致率仅 63% 说明该区域本身噪声高，这是设计上就接受的结果，不是故障。

## 3 BERT 权重缺失时的降级对照

| 权重状态 | audit200 macro-F1 | user300 | comb500 |
|---|---:|---:|---:|
| ep2 与 ep3 齐全 | 0.6769 | 0.6072 | 0.6346 |
| 只有 ep3 | 0.6511 | 0.6006 | 0.6190 |
| 只有 ep2 | 0.6124 | 0.6024 | 0.6042 |
| 无 BERT | 0.5858 | 0.5703 | 0.5748 |

拿这张表可以判断自己的环境装到了哪一档。

## 4 各数字的来源与复现方式

| 数字 | 含义 | 复现方式 |
|---|---|---|
| 0.6161 | fast 口径 audit200 macro-F1 | `eval_all.py --mode frozen` 或 `--mode live` |
| 0.6769 | full 口径 audit200 macro-F1 | 同上，需两个 BERT |
| 0.7400 | full 口径 audit200 准确率 | 同上 |
| 0.5857 / 0.6346 | comb500 macro-F1 | 同上 |
| 全判 other 基线 0.5950 | audit200 中 other 的真实占比 | `eval_all.py --mode all --corpus` 不涉及，用 `data/eval_audit200.csv` 的 label_v3 直接统计 |
| other 占比 0.7106 | 589678 条全语料预测分布 | `eval_all.py --corpus` |
| 漏斗 +1.65 pt | audit200 分歧区 54 条复核后 0.6769 升至 0.6934 | `python eval/eval_funnel_audit200.py` |
| 886 条清洗后 other 47.3% | 标签清洗结果 | 统计 `data/labels_2ch_train_886.csv` 的 label 列 |

## 5 无法用本包复现的部分

诚实说明，以下内容的输入数据不在包内。

| 项目 | 原因 |
|---|---|
| 589,678 条原始语料的抓取 | 原始抓取数据体积过大，且含第三方站点内容。包内只提供打分结果，不提供原文抓取流程 |
| 训练集三塔的推理缓存 | 训练脚本会重新生成，不在包内 |
| 20,000 条蒸馏样本的三塔预测 | 需要先跑全语料 BERT 推理约 7.5 小时，包内只提供蒸馏后的模型 |
| 分歧区 54 条的盲判原文卡片 | 卡片由 `_v22_towers.py` 的上下文索引生成，可重跑，但盲判结论是人工产物，已写死在 `eval_funnel_audit200.py` 的字典里 |

## 6 重训

### 6.1 重训词法塔、蒸馏塔、前文塔

```
python train/train_towers_v22.py
```

需要的工作区相对路径为 `stance3/data/pool_v4.csv`、`stance3/annot/_unify/`、`stance3/data/2ch_scores_v12.csv`。这些路径写死在脚本里，迁移到别的机器需要改开头的 `R`、`D`、`U`、`CACHE` 四个变量。

产出三个 joblib：`tower6_ctx_lex.joblib`、`tower_student_20k.joblib`、`tower1_twitter.joblib`。用时约 15 秒。

### 6.2 重训 BERT

```
python train/train_bert_v21.py
```

数据为 `stance3/data/pool_v4.csv`，产出 `stance3_bert_v21/`。CPU 上 3 轮约 55 分钟。脚本冻结底层 6 层，若换 GPU 建议解冻全部并加大 epoch。

### 6.3 重训的注意事项

一，训练脚本依赖工作区布局，不是自包含的。包内 `train/` 目录是为了让人看懂训练过程，不是为了开箱重跑。

二，重训后 `models/` 下的产物必须用 `eval/eval_all.py --mode live` 重新验证，并更新 `eval/expected_metrics.json`。否则复现判据会失效。

三，随机种子已在脚本中固定，但 sklearn 与 numpy 版本变化仍会导致小数点后第三位漂移。跨版本重训后不要期待与包内数值逐位相同，看量级即可。

## 7 环境

| 组件 | 实测版本 | 用途 |
|---|---|---|
| Python | 3.13.3 | 全部 |
| numpy | 2.5.2 | 全部 |
| scipy | 1.18.0 | 全部 |
| scikit-learn | 1.9.0 | 全部，joblib 由该版本序列化 |
| joblib | 1.5.3 | 全部 |
| pandas | 3.0.5 | 评估与批量处理 |
| torch | 2.15.0+cpu | full 口径 |
| transformers | 5.23.0 | full 口径 |
| sentence-transformers | 6.3.0 | full 口径的 E5 塔 |

sklearn 版本低于 1.9.0 时，`joblib.load` 可能报 InconsistentVersionWarning 或直接失败。这是本包最可能遇到的兼容问题。

## 8 复现失败时的排查顺序

一，先跑 `python selftest.py`，看五行输出是否与 `AGENT_SETUP.md` 第 3 节一致。不一致说明安装有问题，不要继续。

二，再跑 `eval_all.py --mode frozen`。这一条不涉及 BERT，若失败，问题在词法塔、蒸馏塔或融合代码，与 BERT 无关。

三，frozen 通过后再跑 `--mode live`。若 live 失败而 frozen 通过，问题在 BERT 权重或 torch 环境。

四，打印 `StancePredictorV22.load(ROOT).info()` 看 `bert_available`。任一项为 False 就先解决该塔。
