# -*- coding: utf-8 -*-
"""v19 特征管道：2ch 深度清洗 + 短碎片护栏 + 35 维反讽/靶子反转交叉特征"""
import re

import numpy as np
import scipy.sparse as sp

# ---------- 词表 ----------
LEFT_PEJ = r"パヨク|サヨク|ブサヨ|左翼|反日左翼|共産党|お花畑|工作員|売国奴|売国|特亜|支那|チョン|しばき隊|在日"
RIGHT_PEJ = r"ネトウヨ|ウヨ|壺|統一教会|日本会議|アベ信者|高市信者|カルト|戦前回帰|徴兵|保守"
ANTI_REV = r"改憲反対|護憲|9条守|九条守|戦争する国|戦争法案|廃案|改悪|独裁|反対デモ"
PRO_REV = r"改憲賛成|自衛隊明記|国防軍|普通の軍隊|普通の国|9条改正|九条改正|2項削除|二項削除|核武装|スパイ防止法|緊急事態条項|敵基地"
MOCK = r"んでしょ|のかね|思ってんの|本気で思|わけない|笑わせる|頭おかしい|ブーメラン|おめでたい"
HEDGED_NO = (r"急ぐべきではな|急ぐ必要な|時期尚早|慎重に|変える必要な|改正する必要はな|改正いらない|"
             r"今のままで|現行憲法で十分|現行法で対応|議論やってる状況ではな|拙速")
QUOTE_NEWS = (r"と述べ|と報じ|世論調査|アンケート|共同通信|時事通信|毎日新聞|朝日新聞|産経新聞|"
              r"読売新聞|東京新聞|文字起こし|要約")
GREETING = r"おはよう|おつかれ|お疲れ様|よろしく|こんにちは|こんばんは"
POLICY = (r"憲法|9条|九条|改憲|自衛隊|国防|安保|中国|韓国|北朝鮮|ロシア|米軍|核|原発|"
          r"移民|選挙|国会|政党|首相|総理")

RE_ANCHOR = re.compile(r"\[#[a-zA-Z0-9_]+/\d+/\d+\]")
RE_IMG = re.compile(r"\[IMAGE RESOURCES\]")
RE_URL = re.compile(r"h?ttps?://\S+")
RE_REPLY = re.compile(r">>\s*\d[\d,\-]*")
RE_AA = re.compile(r"[^\u3040-\u30ff\u4e00-\u9fff\uFF01-\uFF60\u3000-\u303f0-9A-Za-z]{4,}")
RE_JP = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

CL = re.compile(LEFT_PEJ)
CR = re.compile(RIGHT_PEJ)
CA = re.compile(ANTI_REV)
CP = re.compile(PRO_REV)
CM = re.compile(MOCK)
CH = re.compile(HEDGED_NO)
CQ = re.compile(QUOTE_NEWS)
CG = re.compile(GREETING)
CPOL = re.compile(POLICY)


def clean_2ch(t: str) -> str:
    s = str(t)
    s = RE_ANCHOR.sub(" ", s)
    s = RE_IMG.sub(" ", s)
    s = RE_URL.sub(" ", s)
    s = RE_REPLY.sub(" ", s)
    s = s.replace("<br>", " ").replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    s = RE_AA.sub(" ", s)
    return " ".join(s.split())


def jp_len(s: str) -> int:
    return len(RE_JP.findall(s))


def is_short_fragment(clean: str) -> bool:
    """v3 口径：有效日文字符 < 8 视为无立场碎片"""
    return jp_len(clean) < 8


def is_greeting_only(clean: str) -> bool:
    """纯问候帖：<25字、含问候、不含政策词"""
    return (len(clean) < 25 and bool(CG.search(clean)) and not CPOL.search(clean))


def discourse_features(texts, scale=0.35):
    """35 维显式语篇/反讽交叉特征 -> 稀疏矩阵 (n, 35)"""
    rows = []
    for t in texts:
        s = clean_2ch(t)
        L, R, A, P, M, H, Q = (bool(CL.search(s)), bool(CR.search(s)), bool(CA.search(s)),
                               bool(CP.search(s)), bool(CM.search(s)), bool(CH.search(s)),
                               bool(CQ.search(s)))
        n = jp_len(s)
        f = [
            # 1-8 核心交叉项（治 yes<->no 互翻）
            float(L and A), float(L and M), float(R and P), float(R and M),
            float(A and M), float(P and M),
            float(H),
            float(Q and not L and not R),
            # 9-12 单侧命中
            float(L), float(R), float(A), float(P),
            # 13-16 组合强化
            float(L and A and M), float(R and P and M), float(A and not P), float(P and not A),
            # 17-20 长度/标点
            float(n < 15), float(15 <= n <= 40), float(41 <= n <= 100), float(n > 100),
            float("？" in s or "?" in s), float("！" in s or "!" in s),
            # 23-26 引用/新闻
            float(Q), float(Q and A), float(Q and P), float(s.count(">") >= 2),
            # 27-30 情绪强度
            float(len(re.findall(r"[wｗ]{2,}", s)) > 0), float("笑" in s), float("草" in s),
            float(len(re.findall(r"[！!]{2,}", s)) > 0),
            # 31-35 主体性/判断
            float("私は" in s or "俺は" in s or "自分は" in s),
            float("べき" in s), float("必要" in s), float("反対" in s), float("賛成" in s),
        ]
        rows.append(f)
    return sp.csr_matrix(np.array(rows, dtype=np.float32) * scale)


def apply_guard(texts, probs, L3=("yes", "no", "other")):
    """短碎片 / 纯问候 -> 强制 other"""
    P = np.array(probs, dtype=float)
    oi = L3.index("other")
    for i, t in enumerate(texts):
        c = clean_2ch(t)
        if is_short_fragment(c) or is_greeting_only(c):
            P[i, :] = 0.0
            P[i, oi] = 1.0
    return P
