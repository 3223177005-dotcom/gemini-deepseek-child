# stance3 立场识别：HPC（多卡 GPU）条件下冲击 SOTA（Macro-F1 0.85+）完整工程方案

> **适用环境**：HPC 集群（如 4× NVIDIA A100-80GB / H100，或同等算力节点）
> **核心目标**：突破单机 CPU 词袋/浅层微调的 `0.68` 瓶颈，解决 2ch 论坛特有的**「无口号隐喻」「靶子反转反讽」「省略主语盖楼对喷」**三大难题，将全语料 589,678 条帖子的分类水平推向 **Macro-F1 `0.84 ~ 0.88`、整体准确率 `86% ~ 90%`**（逼近人类双人标注一致性上限 $\kappa \approx 0.86$）。

---

## 一、为什么单机 CPU 会卡在 0.68，而 HPC 能上 0.85+？（四大质变）

| 维度 | 单机 CPU 被迫妥协的做法 | HPC 多卡 GPU 解锁的质变方案 | 解决的核心痛点 |
|---|---|---|---|
| **1. 世界知识与隐喻推理** | `110M BERT-base`（仅维基百科预训练，冻结底 6 层训 3 轮） | **`Qwen2.5-32B-Instruct` / `Swallow-70B` 思维链 LoRA 微调** | 听懂 `「2020露改憲→2023開戦」`、`「俺ら捨て駒になる」` 等跨事件历史类比与反战隐喻 |
| **2. 楼中楼上下文用法** | 输出端对邻居概率求平均（把对喷楼层和闲聊楼层混在一起稀释） | **输入端对话树拼接：`[首楼 #1] + [引用目标 >>n / 前楼 #k-1] + [本帖 #k]`** | 让自注意力机制直接看懂 `「>>12 お花畑すぎて草」` 是在赞同还是反驳第 12 楼 |
| **3. 领域黑话与反讽极性** | 冻结检索型双编码器 E5（导致 `改憲賛成` 与 `改憲反対` 余弦达 `0.943` 极性坍缩） | **58.9 万条 2ch 全语料继续预训练 (DAPT) + 极性反转对比学习 (SupCon)** | 在隐空间把「字面相同、反讽相反」的句对强制推开到余弦 `< 0` |
| **4. 全语料 58.9 万条推理** | 只能跑快筛单塔 + 2万条抽样 | **vLLM 4卡张量并行 + Prefix Caching，1 小时跑完 32B 模型全量 58.9 万条** | 消除 Teacher $\rightarrow$ 词袋 Student 的蒸馏折损，全量享受大模型精度 |

---

## 二、核心工程模块 1：100% 还原 2ch 楼中楼对话树（Input-Level Thread Tree）

在 `2ch_scores_v12.csv` 中，第 1 列 `pid`（如 `2ch/0- (1).seg#107`）完整保留了 **5,729 个讨论串（`seg`）与楼层号（`#k`）**。
在 HPC 上，严禁把帖子当孤立单句输入，必须先运行以下预处理脚本将每条帖子重构为 **「三段式对话树上下文输入」**：

```python
# hpc_step1_build_thread_context.py
import re
import pandas as pd

RE_ANCHOR = re.compile(r">>(\d+)")

def build_contextual_corpus(csv_path: str, out_parquet: str):
    df = pd.read_csv(csv_path, usecols=["pid", "text"])
    df["seg"] = df["pid"].astype(str).str.split("#").str[0]
    df["res_no"] = df["pid"].astype(str).str.split("#").str[1].astype(int)

    # 构建 (seg, res_no) -> text 字典与每个 seg 的首楼主题帖
    seg_res_map = dict(zip(zip(df["seg"], df["res_no"]), df["text"].astype(str)))
    seg_op_map = df.groupby("seg")["text"].first().to_dict()

    ctx_inputs = []
    for row in df.itertuples():
        curr_txt = str(row.text).strip()
        op_txt = str(seg_op_map.get(row.seg, ""))[:120]

        # 优先查找 >>n 显式引用目标；若无，则取紧邻上一楼 #(k-1)
        m = RE_ANCHOR.search(curr_txt)
        target_no = int(m.group(1)) if m else (row.res_no - 1)
        prev_txt = str(seg_res_map.get((row.seg, target_no), ""))[:160]

        formatted = (
            f"【スレ主題 (#1)】: {op_txt}\n"
            f"【直前/引用元レス (#{target_no})】: {prev_txt if prev_txt else '（なし）'}\n"
            f"【判定対象レス (#{row.res_no})】: {curr_txt}"
        )
        ctx_inputs.append(formatted)

    df["ctx_input"] = ctx_inputs
    df.to_parquet(out_parquet, index=False)
```

---

## 三、核心工程模块 2：双轨 SOTA 模型架构（Teacher 32B + Encoder Large）

### 轨道 A（最高上限 Teacher）：`Qwen2.5-32B-Instruct` 思维链（CoT Rationale）LoRA 微调

为什么不直接微调分类头（Sequence Classification），而要用**「1句话潜台词解析 + 标签生成」**？
因为 2ch 的隐喻帖（如 `「露が改憲したのが2020年 戦争始めたのが2023年」`）需要一步逻辑桥接。让 32B 模型在训练时先输出 25 字以内的修辞与逻辑链，再输出最终标签，能激活 LLM 预训练期的推理回路。

#### 1. 训练数据构造（用 Claude/DeepSeek 为现有 3,000 条金标补齐 1 句话解析）
```json
{
  "instruction": "あなたは日本の匿名掲示板(2ch/5ch)の政治スタンス分析の専門家です。RUBRIC_v3基準に従い、【判定対象レス】の憲法改正・国防強化に対する立場を分析し、1行の根拠（修辞・文脈の解釈）に続けて最終ラベル(yes/no/other)を出力せよ。",
  "input": "【スレ主題 (#1)】: ★9条改正しない方が戦争を防げる理由★\n【直前/引用元レス (#577)】: 改正しなければ敵基地攻撃を行えない。\n【判定対象レス (#578)】: まるで植民地のような憲法だったから改正は当然。",
  "output": "根拠: 直前レスの護憲主張に対し、現行憲法を「植民地のような憲法」と否定し9条改正を明確に支持している。\nラベル: yes"
}
```

#### 2. 4×A100 上使用 `LLaMA-Factory` / `trl` 启动 LoRA 微调（耗时 ~25 分钟）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --dataset stance3_cot_gold_v4 \
    --finetuning_type lora \
    --lora_rank 32 --lora_alpha 64 --lora_target all \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 --num_train_epochs 3.0 \
    --bf16 True --Max_len 512 --output_dir ./qwen32b_stance_v22
```

#### 3. 使用 `vLLM` 在 1 小时内跑完 589,678 条全语料（并提取精确对数概率）
在 4 卡 A100 上开启 `enable_prefix_caching=True`（系统提示词 100% 驻留显存 KV Cache，零重复计算），并限制 `max_tokens=45`，设置 `logprobs=5` 直接提取 `yes / no / other` 三个 token 的归一化概率分布：
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="./qwen32b_stance_v22_merged",
    tensor_parallel_size=4,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.92,
    dtype="bfloat16",
)
params = SamplingParams(temperature=0.0, max_tokens=45, logprobs=5)
outputs = llm.generate(all_589k_prompts, params)
```

---

### 轨道 B（毫秒级高吞吐判别模型）：`ku-nlp/deberta-v3-large-japanese`（DAPT + SupCon + Logit-Adjustment）

如果你需要一个不依赖 32B 参数、单卡每秒可推理 2,000 条的独立判别式模型，按以下三步训练：

#### 第 1 步：58.9 万条 2ch 全语料领域自适应继续预训练（DAPT，4×A100 耗时 ~1.5 小时）
- 将 589,678 条 2ch 帖子按讨论串 `seg` 拼接为长文本块，对 `ku-nlp/deberta-v3-large-japanese` 运行 3 个 Epoch 的 Masked Language Modeling（掩码率 `15%`，全词掩码 Whole-Word Masking）。
- **作用**：让 DeBERTa-large 彻底掌握 `パヨク`、`ネトウヨ`、`壺`、`アメポチ`、`お花畑`、`棄民` 等 2ch 专有词汇的上下文共现分布。

#### 第 2 步：极性反转对比学习（Polarity-Inversion SupCon Loss）+ 非对称焦点损失（Logit-Adjusted Focal Loss）
专门解决「错误 A（隐喻漏判）」和「错误 B（反讽互翻）」，总损失函数定义为：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LA-Focal}}(z, y) + 0.25 \cdot \mathcal{L}_{\text{SupCon}}(\mathbf{h}, y)$$

其中 **Logit-Adjusted Focal Loss（确保绝不低估少数派 `no`）** 实现如下：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitAdjustedFocalLoss(nn.Module):
    def __init__(self, class_priors=(0.175, 0.125, 0.700), tau=0.6, gamma_per_class=(2.0, 2.5, 1.0)):
        super().__init__()
        # tau < 0 在训练期惩罚多数类 other 的先验偏置，gamma_no=2.5 重点攻坚难例 no
        self.register_buffer("logit_adj", tau * torch.log(torch.tensor(class_priors, dtype=torch.float32)))
        self.register_buffer("gamma", torch.tensor(gamma_per_class, dtype=torch.float32))

    def forward(self, logits, targets):
        adj_logits = logits + self.logit_adj
        log_probs = F.log_softmax(adj_logits, dim=-1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        gamma_t = self.gamma[targets]
        loss = -((1.0 - pt) ** gamma_t) * log_pt
        return loss.mean()
```

---

## 四、评估体系重建与主动学习闭环（解决 Q3 / Q7）

1. **建立零泄漏的双基准测试集（`Random-700` + `Hard-300`）**：
   - 将已抽好但未标的 `2ch_gold_prop500.xlsx`（500 条无偏比例抽样）标完，与 `audit200_v3_clean`（200 条）合并为 **`Random-700`**（代表真实全语料分布，95% 置信区间收窄至 $\pm 3.1\text{ pt}$）；
   - 将 `user_label300.csv`（300 条旧模型漏判样本）固定为 **`Hard-300`**（专门考核隐喻与反讽召回率）；
   - 训练验证严格采用 **`StratifiedGroupKFold(n_splits=5, groups=seg)`**（同一个讨论串 `seg` 的所有帖子只能在同一个 Fold，严禁跨训练/验证集泄漏！）。
2. **两轮主动学习（Active Learning）精准补齐 1,500 条隐喻难例**：
   - 不要随机标数据！用上述 **轨道 A（`Qwen2.5-32B`）** 与 **轨道 B（`DeBERTa-v3-large`）** 对全语料推理；
   - 专门抽取 **「两模型预测不一致（Disagreement）」或「32B 思维链检测到反讽/历史类比修辞」** 的帖子，用 K-Means 聚成 300 个语义簇，每簇挑 2~3 条（每轮 750 条，共 2 轮 = 1,500 条）交由人工确认并回流训练。

---

## 五、HPC 48 小时首批必跑实验与 2 周冲顶路线图（回答 Q8）

### ⚡ 进场前 48 小时必跑的 3 个决定性实验（立竿见影）

| 实验编号 | 耗时 (4×A100) | 具体内容 | 预期 `Random / Hard` Macro-F1 |
|---|---|---|---:|
| **Exp 1：输入端对话树拼接 + DeBERTa-large 全参微调** | **25 分钟** | 运行 `hpc_step1_build_thread_context.py`，将 `[#1 首楼 + #k-1 前楼 + #k 本帖]` 喂给 `ku-nlp/deberta-v3-large-japanese` 全参微调 5 Epochs（对比单帖输入） | `0.686 / 0.604` $\rightarrow$ **`0.75+ / 0.67+`** |
| **Exp 2：`Qwen2.5-32B-Instruct` 思维链 LoRA 微调** | **35 分钟** | 用现有金标构造「1句话潜台词解析 + 标签」微调 32B 模型，测试其对無口号隐喻帖（如俄乌类比、当炮灰）的召回率 | `0.75+ / 0.67+` $\rightarrow$ **`0.79+ / 0.72+`** |
| **Exp 3：58.9 万条 2ch DAPT 继续预训练** | **1.5 小时** | 在全语料上跑完 3 Epochs MLM 后再执行 Exp 1，并测试 `32B + DeBERTa-DAPT` 双轨融合 | **`0.81+ / 0.75+`** |

### 📅 2 周完整冲顶计划

- **第 1~2 天**：跑完上述 48 小时三大实验，锁定 `Qwen2.5-32B-LoRA + DeBERTa-v3-large-DAPT` 双轨基线（突破 `0.80`）；
- **第 3~6 天**：标完 `gold_prop500` 扩充 `Random-700` 测试集；通过双轨分歧采样完成 1,500 条反讽/隐喻难例的主动学习标注；
- **第 7~10 天**：用扩充后的 ~4,200 条高纯度金标重训 `Qwen2.5-32B-LoRA`，用 vLLM 在 1 小时内刷完 589,678 条全语料软概率分布，并蒸馏出单卡毫秒级推理的终极版 `DeBERTa-v3-large`（最终交付 **Macro-F1 `0.84 ~ 0.87`、`no` 类召回率 `> 75%`、整体准确率 `87%+`**）。
