# -*- coding: utf-8 -*-
"""2ch 同讨论串上下文提取

语料 pid 形如 2ch/0- (1).seg#107，其中 # 前为讨论串标识，后为楼层号。
本模块从带 pid 与 text 的语料表构造每条帖子的前文，供 full 口径使用。

如果手上没有原始语料表，可以不使用本模块，直接调用 predict 即可。
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

RE_QUOTE = re.compile(r">>\s*(\d+)")


def _seg(pid: str) -> str:
    return str(pid).split("#")[0]


def _resno(pid: str) -> int:
    s = str(pid)
    if "#" not in s:
        return 0
    try:
        return int(s.split("#")[1])
    except ValueError:
        return 0


def build_context(pid_list: List[str], text_list: List[str]
                  ) -> Tuple[List[str], List[str]]:
    """返回 (prev_texts, op_texts)，与输入等长。

    prev 取同串被引用楼层（若本帖含 >>n）或紧邻上一楼；op 取同串首楼。
    取不到时返回空字符串。
    """
    if len(pid_list) != len(text_list):
        raise ValueError("pid_list 与 text_list 长度不一致")

    segs = [_seg(p) for p in pid_list]
    first: Dict[str, int] = {}
    for i, s in enumerate(segs):
        first.setdefault(s, i)
    res_map: Dict[Tuple[str, int], int] = {}
    for i, (s, p) in enumerate(zip(segs, pid_list)):
        res_map[(s, _resno(p))] = i

    prev: List[str] = []
    op: List[str] = []
    for i, (s, txt) in enumerate(zip(segs, text_list)):
        op.append(str(text_list[first[s]])[:120])
        j = None
        m = RE_QUOTE.search(str(txt))
        if m:
            j = res_map.get((s, int(m.group(1))))
        if j is None and i - 1 >= 0 and segs[i - 1] == s:
            j = i - 1
        prev.append(str(text_list[j])[:200] if j is not None else "")
    return prev, op
