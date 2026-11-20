# 集成说明

本文档面向自动化代理与其它大模型。目标是读完即可完成配置、验证与调用，不需要其它上下文。

## 0 一页速览

按顺序执行下面的命令。Linux 与 macOS 用第一列，Windows PowerShell 用第二列。

| 步骤 | Linux / macOS | Windows PowerShell |
|---|---|---|
| 建目录 | `mkdir -p stance3_pkg_v22` | `New-Item -ItemType Directory -Force stance3_pkg_v22` |
| 解压主包 | `unzip stance3_pkg_v22_repro.zip -d stance3_pkg_v22` | `Expand-Archive stance3_pkg_v22_repro.zip -DestinationPath stance3_pkg_v22` |
| 装依赖 | `pip install -r stance3_pkg_v22/requirements.txt` | `pip install -r stance3_pkg_v22\requirements.txt` |
| 可选装双 BERT | `unzip stance3_pkg_v22_bert.zip -d stance3_pkg_v22` | `Expand-Archive stance3_pkg_v22_bert.zip -DestinationPath stance3_pkg_v22` |
| 可选装全语料表 | `unzip stance3_pkg_v22_corpus.zip -d stance3_pkg_v22` | `Expand-Archive stance3_pkg_v22_corpus.zip -DestinationPath stance3_pkg_v22` |
| 验证与复现 | `cd stance3_pkg_v22 && python selftest.py && python eval/eval_all.py --mode frozen` | `cd stance3_pkg_v22; python selftest.py; python eval\eval_all.py --mode frozen` |

验证通过的判据见第 3 节。装完后可直接跳到第 10 节拿可运行脚本。

## 1 前置检查

Python 版本下限 3.9，实测版本 3.13.3。执行：

```
python -c "import sys; print(sys.version.split()[0])"
```

磁盘空间：主包 `stance3_pkg_v22_repro.zip` 解压约 45 MB，双 BERT 包 `stance3_pkg_v22_bert.zip` 解压约 850 MB，全语料表包 `stance3_pkg_v22_corpus.zip` 解压约 178 MB。

网络：主包与 fast 口径（以及 `eval_all.py --mode frozen` 复现）完全不需要网络。full 口径的 E5 塔首次运行需要下载或读取缓存 `intfloat/multilingual-e5-small`（约 120 MB）。离线环境下该塔会自动跳过，不影响其它塔。

## 2 安装

第一步，解压主包并安装基础依赖（fast 口径与 frozen 复现到此可用，不需要 torch）：

```
unzip stance3_pkg_v22_repro.zip -d stance3_pkg_v22
pip install -r stance3_pkg_v22/requirements.txt
```

第二步，若需运行 full 口径（六塔全开）及 `eval_all.py --mode live`，安装深度学习依赖并解压双 BERT 权重包：

```
pip install -r stance3_pkg_v22/requirements-full.txt
unzip stance3_pkg_v22_bert.zip -d stance3_pkg_v22
```

解压后必须存在下面两个 BERT 权重目录（各含 `config.json`、`model.safetensors`、`tokenizer_config.json`、`vocab.txt`）：

```
stance3_pkg_v22/bert/stance3_bert_v21/
stance3_pkg_v22/bert/stance3_bert_distill/
```

第三步（可选），若需校验或使用 589,678 条全语料打分结果表，解压语料包：

```
unzip stance3_pkg_v22_corpus.zip -d stance3_pkg_v22
```

## 3 必须通过的验证

```
cd stance3_pkg_v22
python selftest.py
python eval/eval_all.py --mode frozen
```

代理应逐条比对 `selftest.py` 输出中的下面五行：

| 输出内容 | 期望值 | 含义 |
|---|---|---|
| `版本 v22` | 完全一致 | 包版本正确 |
| `标签 ['yes', 'no', 'other']` | 完全一致 | 标签顺序正确，顺序不可调换 |
| `门控阈值 other>=0.55 直接判 other` | 完全一致 | 门控生效 |
| `BERT 权重 ep2=True ep3=True` | 装了 `bert.zip` 时两者均为 True | 未装 `bert.zip` 时两者均为 False |
| `[fast] 命中` | 10/15，其中 easy 5/5 | 低于 10 说明依赖版本异常 |

装了 `stance3_pkg_v22_bert.zip` 后，`[full] 命中` 应为 `12/15`（easy 5/5，hard 7/10）。
接着运行全套复现命令：
- `python eval/eval_all.py --mode frozen`：必须输出 `合计 12 项，通过 12 项，失败 0 项`；
- `python eval/eval_all.py --mode all --corpus`（需已解压 `bert.zip` 和 `corpus.zip`）：必须输出 `合计 28 项，通过 28 项，失败 0 项`；
- `python eval/eval_funnel_audit200.py`：必须输出 `合计 7 项，通过 7 项，失败 0 项`；
- `python train/train_towers_v22.py`：自包含重训测试，耗时约 7 秒，必须复现 `AUD 0.6769 | U300 0.6072 | COMB500 0.6346`。

## 4 文件契约

```
stance3_pkg_v22/
  config.json                    权重、校准系数、门控阈值、评测指标
  requirements.txt               基础依赖（已锁版本）
  requirements-full.txt          深度学习依赖（torch / transformers / sentence-transformers）
  MANIFEST.md                    三包交付清单与组装说明
  README.md                      面向人类的使用说明
  AGENT_SETUP.md                 本文件（面向 GPT/Claude/Gemini 的配置契约）
  REPRODUCE.md                   完整复现指南
  selftest.py                    冒烟测试脚本
  stance3_v22/
    __init__.py                  导出 StancePredictorV22、LABELS、build_context
    predictor.py                 主入口
    context.py                   同讨论串前文提取
    featurizer.py                清洗与 35 维语篇特征
  models/
    tower1_twitter.joblib        6.6 MB（T1 词法塔）
    tower2_e5_twitter.joblib     6.6 MB（T2 E5 混合塔，scale=0.5）
    tower_student_20k.joblib     15.5 MB（T_stu20k 蒸馏塔）
    tower6_ctx_lex.joblib        8.2 MB（T6 前文特征塔）
  bert/                          由 stance3_pkg_v22_bert.zip 解压填入
    stance3_bert_v21/            ep3 权重（424 MB）
    stance3_bert_distill/        ep2 权重（424 MB）
  data/
    eval_audit200.csv            200 条无偏随机评估集（含 prev_text / op_text 列）
    eval_user300.csv             300 条盲区困难评估集（含 prev_text / op_text 列）
    labels_2ch_train_886.csv     886 条清洗后 2ch 人工标注
    train_pool_v4_2721.csv       2721 条训练池（含 prev_text / op_text 列）
    corpus_scores_v22.csv        589678 条全语料打分表（由 corpus.zip 解压填入）
    frozen_probs/                9 个冻结概率文件（供免 BERT 秒级验证）
  eval/
    eval_all.py                  主复现与核验脚本（支持 --mode frozen/live/all --corpus）
    eval_funnel_audit200.py      分歧区级联漏斗复现脚本
    expected_metrics.json        28 项权威验收基准值
  train/
    train_towers_v22.py          自包含词法/前文塔训练与六塔寻优脚本
    train_bert_v21.py            自包含 BERT 微调脚本
  docs/
    RUBRIC_v3.md                 标注规范（含第九章 H1-H5 语篇细则）
    technical_report_v22.md      无 AI 风技术报告
    experiment_log_v21_v22.md    完整实验记录
    hpc_plan.md                  HPC 多卡 GPU 冲击 0.85+ 工程方案
  examples/predict_demo.py
  tests/golden_12.json
```

`models/` 下四个文件缺失任意一个都会导致加载失败。

## 5 API 契约

导入方式。

```python
import sys
sys.path.insert(0, "/abs/path/to/stance3_pkg_v22")
from stance3_v22 import StancePredictorV22, LABELS, build_context
```

`LABELS` 恒为 `["yes", "no", "other"]`，索引 0、1、2 分别对应 yes、no、other，不要重排。

### 5.1 加载

```python
p = StancePredictorV22.load("/abs/path/to/stance3_pkg_v22")
```

参数为包的根目录，不是 `stance3_v22` 子目录。目录内可以没有 `config.json`，此时使用内置默认配置。

### 5.2 推理

```python
predict(texts, mode="fast", prev_texts=None, op_texts=None, return_proba=True) -> list[dict]
```

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| texts | list[str] | 是 | 待判文本，建议先转 str |
| mode | str | 否 | 只能是 `"fast"` 或 `"full"`，其它值抛 ValueError |
| prev_texts | list[str] 或 None | 否 | 同讨论串前文，仅 full 口径使用 |
| op_texts | list[str] 或 None | 否 | 同讨论串首楼，仅 full 口径使用 |
| return_proba | bool | 否 | False 时只返回 label，速度快一点 |

返回值为与输入等长的字典列表。

```python
{"label": "yes", "p_yes": 0.5669, "p_no": 0.3118, "p_other": 0.1213}
```

字段说明。

| 字段 | 类型 | 取值 |
|---|---|---|
| label | str | `"yes"` 或 `"no"` 或 `"other"` |
| p_yes | float | 0 到 1 |
| p_no | float | 0 到 1 |
| p_other | float | 0 到 1 |

三个概率之和为 1，保留四位小数。

`label` 等于三者取最大值的下标。有唯一例外：当 t1 塔单独输出的 other 概率不低于 0.55 时，`label` 强制为 `"other"`，此时 `p_other` 为 0.96，另外两项为 0.02。这是门控规则，不是错误。

### 5.3 只要标签

```python
predict_labels(texts, **kw) -> list[str]
```

### 5.4 查看能力

```python
p.info() -> dict
```

返回字段为 version、labels、gate_other_th、modes、bert_available。`bert_available` 为 `{"ep2": bool, "ep3": bool}`，表示权重与依赖是否同时就绪。

### 5.5 构造前文

```python
build_context(pid_list, text_list) -> (prev_texts, op_texts)
```

两个入参等长，返回两个等长的列表。

pid 格式为 `2ch/<board> <thread>.seg#<楼层号>`。`#` 之前为讨论串标识，同一讨论串的帖子必须在列表中连续排列，否则前文会错乱。

若本帖含 `>>数字` 的引用，优先取被引用楼层作为前文；否则取紧邻上一楼；都取不到时返回空字符串。

### 5.6 异常

| 情况 | 行为 |
|---|---|
| texts 为空列表 | 返回空列表 |
| mode 非法 | 抛 ValueError |
| BERT 权重缺失 | 该塔用 t1 结果顶替，不抛异常 |
| transformers 或 torch 未装 | 同上，不抛异常 |
| sentence-transformers 未装或离线 | E5 塔用 t1 结果顶替，不抛异常 |
| models 下 joblib 缺失 | 抛 FileNotFoundError |

## 6 config.json 字段

| 字段 | 类型 | 合法值 | 改动影响 |
|---|---|---|---|
| version | str | `"v22"` | 仅标识，改它无功能影响 |
| labels | list[str] | 固定三元素且顺序不可变 | 改动会破坏所有塔的概率对齐 |
| gate_other_th | float | 0 到 1，默认 0.55 | 调低会减少 other 判出量，调高会增加 |
| modes.fast.towers | dict | t1 与 stu 权重，和为 1 | 调高 t1 占比会降低立场召回 |
| modes.fast.mult | null 或 list[float] | fast 口径必须为 null | 填值会造成双重加权，实测精度自 0.5858 跌至 0.5571 |
| modes.full.towers | dict | 六个权重，代码内会归一化 | 已按网格搜索确定，不建议手改 |
| modes.full.mult | list[float] | 三项，实测 1.32 1.42 1.20 | 用于抵消训练期类别权重引入的偏置 |
| bert.ep3 | str | 相对包根的权重目录 | 改路径即换模型 |
| bert.ep2 | str | 同上 | 缺失时自动退化 |
| bert_maxlen | int | 96 | 调大更慢，语义内容极少超长 |
| e5_model | str | HF 模型名 | 换模型需重新训练对应塔，只改此处无效 |
| e5_scale | float | 0.5 | 改它会使 t2 塔与训练时不一致，精度下降 |

`metrics` 与 `accuracy` 与 `corpus_distribution` 三个字段仅作记录，代码不读取。

## 7 精度分级

audit200 为从 589,678 条语料随机抽取的 200 条，代表真实分布。user300 为专挑旧模型漏判样本构造的困难集，不代表分布，不可用于对外报告分布类结论。

| BERT 权重状态 | audit200 macro-F1 | audit200 准确率 | COMB500 macro-F1 |
|---|---:|---:|---:|
| ep2 与 ep3 齐全 | 0.6769 | 74.0% | 0.6346 |
| 只有 ep3 | 0.6511 | 72.0% | 0.6190 |
| 只有 ep2 | 0.6124 | 69.0% | 0.6042 |
| 无 BERT，fast 口径 | 0.5858 | 67.0% | 0.5748 |

audit200 上全判 other 的基线准确率为 59.5%。

分项成绩（full 口径，ep3 在位的配置，audit200）：

| 类别 | 条数 | 召回率 | 精确率 |
|---|---:|---:|---:|
| yes | 47 | 53.2% | 59.5% |
| no | 34 | 61.8% | 65.6% |
| other | 119 | 85.7% | 81.0% |

## 8 故障对照

| 现象 | 原因 | 处理 |
|---|---|---|
| ModuleNotFoundError: sklearn | 未装依赖 | `pip install -r requirements.txt` |
| FileNotFoundError: tower1_twitter.joblib | models 目录未解压或路径传错 | 确认 load 的入参是包含 models 的那层目录 |
| `BERT 权重 ep3=False` 但文件已放 | transformers 或 torch 未装 | 装完后重跑，`_bert_ready` 会同时检查依赖 |
| selftest 中 `[fast] 命中` 低于 10 | numpy 或 sklearn 版本过旧导致概率漂移 | 升级到 numpy 2.x 与 sklearn 1.x |
| 首次 full 推理卡住数分钟 | 正在下载 E5 模型 | 等待，或设 `HF_ENDPOINT` 镜像，或直接改用 fast |
| SSL 或 Hub 连接错误 | 离线环境 | 不影响 fast，full 的 E5 塔会自动跳过 |
| 内存占用过高 | 一次传入过多文本 | 按 4 万条分块，见第 10 节 |
| Windows 控制台乱码 | 编码非 UTF-8 | 设 `PYTHONIOENCODING=utf-8` |

## 9 硬约束

代理在配置与调用过程中不要做下面几件事。

一，不要修改 `models/` 下任何 `.joblib`。这些文件由训练脚本产出，手工改动会使概率分布失效。

二，不要在 fast 口径上填 `modes.fast.mult`。fast 口径的两个塔在训练期已带类别权重，再乘校准系数会双重加权，实测精度自 0.5858 跌至 0.5571。

三，不要把 user300 的成绩当作真实分布下的准确率对外陈述。该集合 100% 抽自旧模型漏判样本。

四，不要用全量预测的 other 占比描述语料真实构成。全量为 71.06%，随机样本的真实值是 59.5%，差 11.5 个百分点，原因是立场召回率只有 53% 到 62%。

五，不要在 audit200 上做任何调参。该集合是主评估集，调参会使报出的数字失效。

六，`labels` 顺序固定为 yes、no、other。任何重排都会使概率与标签错位，且不会报错。

## 10 最小可运行脚本

逐字复制即可运行，只需把第一行的路径换成实际解压路径。

```python
import sys, collections
sys.path.insert(0, "/abs/path/to/stance3_pkg_v22")
from stance3_v22 import StancePredictorV22, build_context

ROOT = "/abs/path/to/stance3_pkg_v22"
p = StancePredictorV22.load(ROOT)
print(p.info())

# 三条示例，其中后两条带立场
texts = [
    "改憲賛成。9条に自衛隊を明記すべきだ。",
    "憲法9条を守れ。改憲反対。",
    "おはようございます",
]
for t, r in zip(texts, p.predict(texts, mode="fast")):
    print(r["label"], r["p_yes"], r["p_no"], r["p_other"], t[:30])

# 带前文的 full 调用
pids = ["2ch/newsplus/1.seg#1", "2ch/newsplus/1.seg#2"]
bodies = ["9条改正の是非を議論するスレ", ">>1 改正は当然だろ"]
prev, op = build_context(pids, bodies)
print("prev =", prev)
print("op   =", op)
print(p.predict(bodies, mode="full", prev_texts=prev, op_texts=op))

# 三类占比
cnt = collections.Counter(p.predict_labels(texts))
n = sum(cnt.values())
print({k: round(cnt.get(k, 0) / n * 100, 1) for k in ("other", "yes", "no")})
```

### 批量处理模板

```python
import pandas as pd
from stance3_v22 import StancePredictorV22

p = StancePredictorV22.load(ROOT)
df = pd.read_csv("input.csv")            # 必须含 text 列
CHUNK = 40000
rows = []
for i in range(0, len(df), CHUNK):
    part = df["text"].iloc[i:i + CHUNK].astype(str).tolist()
    for t, r in zip(part, p.predict(part, mode="full")):
        rows.append({"text": t, **r})
pd.DataFrame(rows).to_csv("output.csv", index=False, encoding="utf-8-sig")
```

## 11 环境与版本

本包在下列环境实测通过。

| 组件 | 版本 | 必需 |
|---|---|---|
| Python | 3.13.3 | 是，下限 3.9 |
| numpy | 2.5.2 | 是 |
| scipy | 1.18.0 | 是 |
| scikit-learn | 1.9.0 | 是 |
| joblib | 1.5.3 | 是 |
| pandas | 3.0.5 | 仅批量处理示例使用 |
| torch | 2.15.0+cpu | 仅 full 口径 |
| transformers | 5.23.0 | 仅 full 口径 |
| sentence-transformers | 6.3.0 | 仅 full 口径的 E5 塔 |

模型文件由 scikit-learn 产出，建议使用表中版本或更高。跨大版本升级后应先跑第 3 节验证，确认 `[fast] 命中` 仍为 10/15。

## 12 速度参考

本机为 18 线程 CPU，无 GPU。下表为实测值。

| 口径 | 速度 | 589,678 条全量耗时 |
|---|---|---|
| fast，predict | 3,006 条每秒 | 约 3.3 分钟 |
| fast，predict_labels | 3,155 条每秒 | 约 3.1 分钟 |
| full，含 BERT | 约 36 条每秒 | 约 7.5 小时 |

测试条件为单进程、文本取自真实语料前 5,000 条、已预热。GPU 环境下 full 会显著加快，但本包未针对 GPU 做批处理优化，需自行调整 `predictor.py` 中 `_bert_proba` 的批大小与设备放置。

## 13 任务定义

供代理判断是否适用，不适用于其它议题。

| 标签 | 定义 |
|---|---|
| yes | 支持日本政府推进的改宪与国防强化议题，含自卫队明记、九条改正、紧急事态条项、防卫费增额、核武装、间谍防止法 |
| no | 反对上述议题，或支持政府所反对的议题，含移民扩大、夫妇别姓、同性婚 |
| other | 无立场，含转述、纯局势分析、情绪辱骂、杂谈、碎片发言 |

已知短板：不含任何口号词的隐喻与反讽帖会被漏判成 other，该类错误占全部错误的 46%。若下游任务对这类文本敏感，需要人工复核 `divergent_zone` 标记的样本，或改用更大模型，详见工作区内的 HPC 方案文档。
