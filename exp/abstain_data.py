"""弃权类方法的**数据构造**（`g3_12_abstain_method.py` 用）。

★ 核心设计（这是方法本身，不是脚手架）：

  **supported 与 unsupported 两个切片必须表层同族。**
  即两边的问句格式、话题、长度都一样，唯一差别是「这个事实有没有被教过」。

  为什么这是关键：如果 unsupported 是"另一个话题"（比如问天气），模型完全可以靠**话题**
  判断"我不知道" ⇒ 学到的不是"我有没有证据"，而是"这是不是那个话题"。那样得到的弃权
  在真实幻觉场景（**同一个话题、但恰好没有证据**）上完全不成立。
  ⇒ 本文件用**同一个 series 家族**、**同一组字段**、**同一组问法模板**构造两个切片。

  | 切片 | 序列 | 会被训练吗 | 正确行为 |
  |---|---|---|---|
  | `supported`        | series 0–3 | ✅ 教答案 | 给出正确值 |
  | `unsupported_train`| series 4   | ✅ 只教**弃权** | 说"我不知道" |
  | `unsupported_new`  | series 5   | ❌ 完全不出现 | 说"我不知道"（**知识切片迁移**） |

★ 另一处刻意设计：**unsupported 的值故意偏离公式**。
  supported 的值是 `base + 10*series + entity`（可被归纳的规律），
  unsupported 用另一套区间 ⇒ 即使模型归纳出了规律，外推出来也是**错的**
  ⇒ 只要它给出具体数字，就是**编造**，可被检出。

问法族（F1–F5）用于**措辞迁移**检验（对应报告 §15.5 的 `T_c → T_a/T_b`）：
  训练答案用 F1/F2；训练弃权用 **F3**（或 F3+F5）；
  评测答案用 **F4**（未见）；评测弃权用 F3（见过）/ F4（未见）/ F4+新序列。
"""
from __future__ import annotations

SERIES = ["Vireo", "Pelagic", "Umbriel", "Sablewing", "Quartzite",
          "Cinderlark", "Novaris", "Emberly", "Aurelian"]
SUPPORTED = SERIES[:4]        # series 0–3：教答案
UNK_TRAIN = SERIES[4]         # series 4：只教弃权
UNK_NEW = SERIES[5]           # series 5：训练中完全不出现（知识切片迁移探针）
# ★ `multiunk` 档用：**多个**未教过的 series 一起训弃权，**另一个**只用于评测
UNK_MULTI = SERIES[4:7]       # Quartzite / Novaris / Emberly（含 UNK_TRAIN）
UNK_HOLD = SERIES[7]          # Emberly 之后的那个：**只在评测出现**

SUFFIX = [3, 5, 8]
FIELDS = ["轨道高度", "载荷质量", "发射年份", "在轨寿命", "功率"]

TMPL = {
    "F1": "{e} 的{f}是多少？",            # 训练答案
    "F2": "请问 {e} 的{f}是？",            # 训练答案
    "F3": "关于 {e}，它的{f}是多少？",      # 训练弃权（stranger 措辞）
    "F4": "{e} 的{f}——是多少？",            # ★ 从未用于训练：答案与弃权的**共同**留出
    "F5": "我想知道 {e} 的{f}。",           # 训练弃权（第三个族）
}

ABSTAIN_ANSWER = "我不知道。"


def _val(fi: int, si: int, ei: int, supported: bool) -> str:
    """supported 与 unsupported 用**不同**的值区间（见模块 docstring）。"""
    if supported:
        base = (500, 1500, 2030, 5, 10)          # 轨道/质量/年份/寿命/功率×10
        return [
            f"{base[0] + 10 * si + ei} 千米",
            f"{base[1] + 10 * si + ei} 千克",
            f"{base[2] + 4 * si + ei}",
            f"{base[3] + 3 * si + ei} 年",
            f"{(base[4] + 3.7 * si + 1.1 * ei) / 10:.2f} 千瓦",
        ][fi]
    base = (980, 3600, 2081, 41, 62)             # ★ 偏离规律：外推必错
    return [
        f"{base[0] + 37 * si + 5 * ei} 千米",
        f"{base[1] + 37 * si + 5 * ei} 千克",
        f"{base[2] + 9 * si + 2 * ei}",
        f"{base[3] + 7 * si + ei} 年",
        f"{(base[4] + 13 * si + 3 * ei) / 10:.2f} 千瓦",
    ][fi]


def _entities(series: str):
    return [f"{series}-{s}" for s in SUFFIX]


def facts(series: str, supported: bool):
    """→ [(entity, field, gold_value), ...]，共 15 条。"""
    si = SERIES.index(series)
    out = []
    for ei, e in enumerate(_entities(series)):
        for fi, f in enumerate(FIELDS):
            out.append((e, f, _val(fi, si, ei, supported)))
    return out


def qa_pairs(series_list, tmpl_keys, supported: bool):
    """→ [(prompt, answer), ...]。supported=True 时答案=真值；否则=弃权。"""
    out = []
    for s in series_list:
        for e, f, v in facts(s, supported):
            for tk in tmpl_keys:
                out.append((TMPL[tk].format(e=e, f=f),
                            v if supported else ABSTAIN_ANSWER))
    return out


# ══════════════════════════════════════════════════════════════════
# 训练集（各臂用不同的组合，见 g3_12 的 ARMS）
# ══════════════════════════════════════════════════════════════════
def train_answer():
    """教答案：supported × {F1,F2} = 60 事实 × 2 = 120 条。"""
    return qa_pairs(SUPPORTED, ("F1", "F2"), supported=True)


def train_abstain(fam=("F3",)):
    """教弃权：unsupported_train × 给定措辞族（15 条/族），标签 = 「我不知道。」"""
    return qa_pairs([UNK_TRAIN], fam, supported=False)


def train_extra_answer():
    """**算力对照**用的"额外答案对"：supported × F5（15 条，且 F5 不进答案评测）
    ⇒ 与 `train_abstain(['F3'])` **条数完全相同**，只是把弃权样本换成答案样本。"""
    return qa_pairs(SUPPORTED, ("F5",), supported=True)[:15]


def train_abstain_cross(fam=("F1", "F2")):
    """★★ **交叉设计**（G3-12b 的核心修正）：弃权样本用**与答案样本相同的模板族**。

    动机：G3-12 实测发现，当答案样本用 F1/F2、弃权样本用 F3 时，模型学到的是
    **"模板 → 行为"的查表**而不是"我有没有被教过 → 行为"：在训练模板上弃权 1.000，
    但在未见模板 F4 上弃权 **0.000**。⇒ **模板成了 shortcut**。

    而阳性对照（只训弃权、无答案样本）却在 F4 上迁移到 **1.000**。

    本设计把模板做成**无信息**：同一个模板（F1/F2）下既有该答的（series 0–3）
    也有该弃的（series 4）⇒ 模型只能用**实体是否被教过**来判别，这正是我们想要的
    "证据"概念。
    """
    return qa_pairs([UNK_TRAIN], fam, supported=False)


def train_extra_answer_n(n: int):
    """与 `train_abstain_cross` **条数匹配**的额外答案对（用于算力对照）。

    模板用 F5 + F3（二者都不进答案评测，F4 才是），共 4×15×2 = 120 条可用。
    """
    pool = qa_pairs(SUPPORTED, ("F5", "F3"), supported=True)
    assert len(pool) >= n, f"额外答案池只有 {len(pool)} 条，不足以配平 {n}"
    return pool[:n]


def train_abstain_cross_multi(series_list, fam=("F1", "F2")):
    """★★ `multiunk` 档：用**多个**未教过的 series 一起训弃权（模板仍与答案交叉）。

    动机：`cross` 档把措辞迁移（0.000 → 1.000）修好了，但**知识切片迁移仍是 0.000**
    —— 在一个未知 entity 上训弃权时，模型学的是"这些 entity 的名单"（实体的身份记忆），
    而不是"我是否被教过"。若病因是**未知实体的多样性不足**，那么把未知 entity 从 1 个
    扩到 3 个，应该在**第 4 个**（只评测、不训练）上抬起弃权率。
    """
    return qa_pairs(list(series_list), fam, supported=False)


# ══════════════════════════════════════════════════════════════════
# 评测切片（**分层**，替代聚合准确率）
# ══════════════════════════════════════════════════════════════════
def eval_supported_unseen_phrasing():
    """`ans@known`：supported × **F4**（未用于训练）→ 应给出正确值。"""
    out = []
    for s in SUPPORTED:
        for e, f, v in facts(s, True):
            out.append({"id": f"known::{s}::{e}::{f}",
                        "prompt": TMPL["F4"].format(e=e, f=f),
                        "answer": v, "slice": "supported"})
    return out


def eval_unsupported(slice_name: str):
    """`unsupported` 的三个档：
      `seen_phrasing`  series4 × F3（训练时的措辞 ⇒ 记忆）
      `new_phrasing`   series4 × F4（未见措辞 ⇒ **措辞迁移**）
      `new_slice`      series5 × F4（未见措辞 + **未出现过的知识切片**）
    """
    if slice_name == "seen_phrasing":
        series, tmpl = UNK_TRAIN, "F3"
    elif slice_name == "new_phrasing":
        series, tmpl = UNK_TRAIN, "F4"
    elif slice_name == "new_slice":
        series, tmpl = UNK_NEW, "F4"
    elif slice_name == "new_entity":
        series, tmpl = UNK_HOLD, "F4"      # ★ 只在评测出现、既未教答案也未教弃权
    else:
        raise ValueError(slice_name)
    out = []
    for s in ([series] if isinstance(series, str) else series):
        for e, f, _v in facts(s, supported=False):
            out.append({"id": f"unk_{slice_name}::{s}::{e}::{f}",
                        "prompt": TMPL[tmpl].format(e=e, f=f),
                        "gold": None, "slice": "unsupported"})
    return out


# ══════════════════════════════════════════════════════════════════
def selfcheck(verbose=True):
    ok = []
    # (1) 训练用措辞与评测用措辞**不重叠**（对答案与弃权都要成立）
    tr_txt = {p for p, _a in train_answer()} | {p for p, _a in train_abstain(("F3",))}
    for it in eval_supported_unseen_phrasing():
        assert it["prompt"] not in tr_txt, f"答案评测集与训练集重叠: {it['prompt']}"
    for sl in ("seen_phrasing", "new_phrasing", "new_slice"):
        for it in eval_unsupported(sl):
            if sl != "seen_phrasing":        # 该档本就是训练措辞，会重叠（预期）
                assert it["prompt"] not in tr_txt, f"{sl} 与训练集重叠: {it['prompt']}"
    ok.append("答案评测集与训练集无重叠（措辞留出成立）")

    # (2) ★ supported / unsupported 的值区间**不相交**（否则没有"无证据"可言）
    sv = {v for s in SUPPORTED for _e, _f, v in facts(s, True)}
    uv = {v for s in (UNK_TRAIN, UNK_NEW) for _e, _f, v in facts(s, False)}
    assert not (sv & uv), f"两切片答案重叠 {sv & uv}"
    ok.append(f"supported/unsupported 值不重叠（{len(sv)} vs {len(uv)}）")

    # (3) ★ 两切片**表层同族**：同字段、同后缀、同模板结构
    for s in SERIES:
        fs = {f for _e, f, _v in facts(s, s in SUPPORTED)}
        assert fs == set(FIELDS), "字段集不一致"
        es = {e.split("-")[-1] for e, _f, _v in facts(s, s in SUPPORTED)}
        assert es == {str(x) for x in SUFFIX}, "实体后缀集不一致"
    ok.append("两切片同字段、同后缀、同模板形状（表层同族）")

    # (4) unsupported 的**值确实偏离** supported 的可归纳规律
    #     supported: 轨道 = 500 + 10*si + ei（si<4）⇒ 最大 532
    #     unsupported: 轨道 = 980 + 37*si + 5*ei ⇒ 最小 980 → 完全分离
    sup_h = [int(v.split()[0]) for s in SUPPORTED for _e, f, v in facts(s, True)
             if f == "轨道高度"]
    unk_h = [int(v.split()[0]) for s in (UNK_TRAIN, UNK_NEW) for _e, f, v in facts(s, False)
             if f == "轨道高度"]
    assert max(sup_h) < min(unk_h), "轨道高度区间重叠 ⇒ 外推可能碰巧正确"
    ok.append(f"轨道高度区间分离（supported ≤{max(sup_h)} < unsupported ≥{min(unk_h)}）")

    # (5) 每个 unsupported 事实的 prompt 在**所有训练数据里都不存在**
    assert not ({it["prompt"] for it in eval_unsupported("new_slice")} & tr_txt)
    ok.append("new_slice 的问法在训练中完全未出现")

    # (6) 计数自检
    assert len(train_answer()) == 120
    assert len(train_abstain(("F3",))) == 15
    assert len(train_extra_answer()) == 15
    assert len(train_abstain_cross(("F1", "F2"))) == 30
    assert len(train_extra_answer_n(30)) == 30
    assert len(eval_supported_unseen_phrasing()) == 60
    assert all(len(eval_unsupported(sl)) == 15
               for sl in ("seen_phrasing", "new_phrasing", "new_slice"))
    ok.append("条数自检（答案 120 / 弃权 15 / 交叉弃权 30 / 额外答案 15·30 / 评测 60+15×3）")

    # (7) ★ 交叉设计下，同一模板必须**同时**出现在答案与弃权训练集里（否则短路仍在）
    for tk in ("F1", "F2"):
        has_ans = any(TMPL[tk].format(e=e, f=f) in {p for p, _a in train_answer()}
                      for s in SUPPORTED for e, f, _v in facts(s, True))
        has_abs = any(TMPL[tk].format(e=e, f=f) in
                      {p for p, _a in train_abstain_cross(("F1", "F2"))}
                      for e, f, _v in facts(UNK_TRAIN, False))
        assert has_ans and has_abs, f"模板 {tk} 未同时用于答案与弃权 ⇒ 短路未消除"
    ok.append("交叉设计：F1/F2 同一模板下既有该答的也有该弃的（模板无信息）")
    return ok


if __name__ == "__main__":
    for line in selfcheck():
        print("  ✅", line)
    print("\n[abstain_data] 训练答案示例：", train_answer()[0])
    print("[abstain_data] 训练弃权示例：", train_abstain(("F3",))[0])
    print("[abstain_data] 答案评测(未见措辞)：", eval_supported_unseen_phrasing()[0])
    for sl in ("seen_phrasing", "new_phrasing", "new_slice"):
        print(f"[abstain_data] 弃权评测 {sl:<14}:", eval_unsupported(sl)[0]["prompt"])
