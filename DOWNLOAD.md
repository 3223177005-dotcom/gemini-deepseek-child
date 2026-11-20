# 大文件下载

仓库本体只含代码、文档、评估数据与四个词法塔（约 39 MB）。两个大文件作为 Release 附件提供。

## 必需，跑 full 口径时下载

`stance3_pkg_v22_bert.zip`（790 MB）

含两套日语 BERT 微调权重，解压后放到仓库根目录，形成 `bert/stance3_bert_v21/` 与 `bert/stance3_bert_distill/`。

```
unzip stance3_pkg_v22_bert.zip -d .
```

不下载时 fast 口径仍可运行，只是 full 口径会退化为 fast 成绩。

## 可选，需要现成全语料结果时下载

`stance3_pkg_v22_corpus.zip`（52 MB）

含 589,678 条 2ch 全语料打分表，解压后为 `data/corpus_scores_v22.csv`。

```
unzip stance3_pkg_v22_corpus.zip -d .
```

## 下载命令

```
gh release download v22.0 --repo 3223177005-dotcom/gemini-deepseek-child --pattern "*.zip"
```

或直接打开 Release 页面

```
https://github.com/3223177005-dotcom/gemini-deepseek-child/releases/tag/v22.0
```

## 校验

下载后跑下面的命令，应输出 `合计 12 项，通过 12 项，失败 0 项`。

```
pip install -r requirements.txt
python eval/eval_all.py --mode frozen
```

装好 BERT 后再跑 `python eval/eval_all.py --mode all --corpus`，应输出 `合计 28 项，通过 28 项，失败 0 项`。
