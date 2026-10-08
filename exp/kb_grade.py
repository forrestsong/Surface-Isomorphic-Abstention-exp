"""真实知识边界数据集的**加固判分器**（`g3_14` / `g3_16` 共用）。

★ 起因：`capability.grade_exact` 在这个数据集上有**三处**失效（报告 §26.1，实测）：

  ① **生成预算**（不是判分器的问题，但同属"仪表"）：`gen(mt=16)` 会把 base 的输出腰斩。
     base 的风格是"先铺背景"，实测响应 18–38 字符、全部在 16 token 处截断：
       `'列奥纳多·达·芬奇（Leonardo da Vinci）一生'` → 在说出「蒙娜丽莎」前就断了。
     ⇒ `ans@known` 同时测"知不知道"和"**说话简洁不简洁**"。
     实测：base 0.692(mt16) → **0.885(mt40)**，**符号从 +0.038 翻成 −0.154**。

  ② **金标是实体名的子串 ⇒ 假阳性**：`蒙娜丽莎::画中人物的身份`（gold `丽莎`）。base 答
     `'…目前**历史上并没有定'`（其实在**回避**，也就是弃权），但响应里含 `蒙娜丽莎`
     ⇒ 子串匹配命中。同类：`哈姆雷特::丹麦王子`（gold = 实体名本身）。
     ⇒ 与 §18 的 **ASCII-only 身份匹配**同型（**判据的子串范围选错**）。
     ⇒ 这类条目**无法用子串判分**（"模型在陈述实体名"与"模型在回答"不可区分）
     ⇒ 定名为**病态条目**，由 `is_illposed` 剔除。

  ③ **中文数字 ⇒ 假阴性**：`日本::主要岛的数量` gold `4`，base 答
     `'日本列岛主要由**四大岛**组成'` ⇒ `grade_exact` 的数字分支只认 Arabic 数字 ⇒ 判错。
     模型其实答对了。

★★ `grade_hard` 的实体名消除有一个**必须做对**的细节：
   只能移除**不与金标重叠**的实体名出现。否则 `巴西::首都`（gold `巴西利亚` ⊃ 实体 `巴西`）
   会被误伤成 `▢利亚` ⇒ 把本来对的判成错（这是本模块第一版实测到的自伤）。
"""
from __future__ import annotations

import re

import capability as cap

CJK_NUM = {"零": "0", "〇": "0", "一": "1", "二": "2", "两": "2", "三": "3",
           "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
NUM_CHARS = "".join(CJK_NUM) + "0123456789"


def norm_key(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"[\s，。、：:；;！!？?“”\"'（）()\[\]]+", "", s)


def is_illposed(it) -> bool:
    """金标是**实体名或属性名**的子串 ⇒ 子串匹配必然假阳性，无法判分。"""
    g = norm_key(it.get("gold") or "")
    return bool(g) and (g in norm_key(it.get("entity"))
                        or g in norm_key(it.get("attr")))


def _strip_entity(resp: str, entity: str, gold: str) -> str:
    """移除实体名回显，但**保留与金标重叠的**那些出现（否则会自伤，见模块 docstring）。"""
    if not entity:
        return resp
    spans = [(m.start(), m.end()) for m in re.finditer(re.escape(gold), resp)] \
        if gold else []

    def overlaps(s, e):
        return any(not (e <= a or s >= b) for a, b in spans)

    out, last = [], 0
    for m in re.finditer(re.escape(entity), resp):
        if overlaps(m.start(), m.end()):
            continue
        out.append(resp[last:m.start()])
        out.append("▢")
        last = m.end()
    out.append(resp[last:])
    return "".join(out)


def grade_hard(resp: str, it) -> bool:
    """加固判分：① 去实体名回显（保金标重叠）② 中文数字感知 ③ 病态条目由调用方剔除。"""
    gold = it.get("gold")
    if gold is None:
        return False
    r = _strip_entity(resp, it.get("entity", ""), gold)
    # ② 纯阿拉伯数字金标（≤2 位）：允许等价的中文数字；该中文字不能紧跟另一个数字字符
    if gold.isdigit() and len(gold) <= 2:
        for ch, d in CJK_NUM.items():
            if d == gold:
                for m in re.finditer(re.escape(ch), r):
                    prev = r[m.start() - 1] if m.start() > 0 else ""
                    if prev not in NUM_CHARS:
                        return True
    if cap.grade_exact(r, gold):
        return True
    return cap.grade_exact(r.replace(" ", ""), gold.replace(" ", ""))
