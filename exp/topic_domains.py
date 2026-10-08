"""N 个**同构**虚构域 + 一条**单轴**的「线索可用性」阶梯（`g3_9_missing_cue.py` 用）。

动机（来自 `奇点实验报告.md` §12.4 的自评）：
    §12.4 已证「**词汇重叠**」不是门控的威胁（P1/P2/P3 三对全可分），
    并明确写下：**主要风险已从「线索被稀释」转移到「线索缺失/歧义」**，
    且真正未测的是 ①线索缺失（指代）②概率性主题线索 ③多域规模 ④表述漂移。
    本文件就是为了测①③（以及②的极端形式：线索可靠性 = 0）。

★ 单轴设计：所有域**同构**（同样的 5 个字段、同样的 3 个实体后缀、同样的问法模板），
  唯一的差别是 **series 名**（= 判别线索）。于是「线索可用性」可以沿**一条轴**
  严格递减，而不掺杂其它变量：

  ┌ `cue`      完整线索     "Vireo-3 的轨道高度是多少？"        ← 阳性对照（必须可分）
  ├ `partial`  部分线索     "3 号的轨道高度是多少？"            ← 后缀共享于全部 N 域 ⇒ 歧义
  ├ `generic`  无线索       "该航天器的轨道高度是多少？"         ← 信息量为 0（可识别性下限）
  └ `coref`    线索在上一轮 "Vireo-3 是什么？" + "那它的轨道高度是多少？"
                  ⇒ 当前轮无线索；只有**多轮上下文**里才有 ⇒ 检验「上下文能否救回路由」

★ 为什么这是**公平**而不是「无解设计」：
   `generic`/`partial` 在**单轮**里是**信息论上不可识别**的 —— 任何读出都做不到。
   所以它们**不是**「门控失败」，而是在问一个**不同的问题**：
     **门控能不能知道自己不知道？**（⇒ 弃权/abstain，而不是猜）
   这正是可部署系统真正需要的性质。`coref` 则问：多轮上下文是否**必要且充分**地
   补回判别信息（用 full-context 特征 vs 仅当前轮特征对照）。

★ 域间答案**全局唯一**（脚本内断言）⇒ 错路由一定产出可检出的错误答案，
   不会出现「答对了但路由错了」的假象。
"""
from __future__ import annotations

# ── N=8 个 series（判别线索）。全部是**生造词**，不与真实实体撞车。 ──────
SERIES = ["Vireo", "Pelagic", "Umbriel", "Sablewing",
          "Quartzite", "Emberly", "Novaris", "Cinderlark"]

# 实体后缀**跨域共享** ⇒ `partial` 档（"3 号"）在 N 个域间完全歧义。
SUFFIX = [3, 5, 8]

# 字段**跨域全同** ⇒ 字段名不携带任何域信息（这是关键的单轴控制）。
FIELDS = ["轨道高度", "载荷质量", "发射年份", "在轨寿命", "功率"]


def _val(fi: int, di: int, ei: int) -> str:
    """(字段, 域序号, 实体序号) → 答案字符串。

    设计：域序 `di` 是**十位量级**的偏移，实体序 `ei` 是个位量级的偏移
    ⇒ 同字段内 (di, ei) 唯一 ⇒ **全部答案全局唯一**（见自检断言）。
    """
    if fi == 0:
        return f"{500 + 10 * di + ei} 千米"
    if fi == 1:
        return f"{1500 + 10 * di + ei} 千克"
    if fi == 2:
        return f"{2030 + 4 * di + ei}"
    if fi == 3:
        return f"{5 + 3 * di + ei} 年"
    return f"{1.0 + 0.37 * di + 0.11 * ei:.2f} 千瓦"


def _build():
    doms = {}
    for di, s in enumerate(SERIES):
        name = s.lower()
        ents = {}
        for ei, sfx in enumerate(SUFFIX):
            ents[f"{s}-{sfx}"] = {FIELDS[fi]: _val(fi, di, ei)
                                  for fi in range(len(FIELDS))}
        doms[name] = {"series": s, "title": f"{s} 系列航天器", "entities": ents}
    return doms


DOMAINS = _build()
DOM_NAMES = list(DOMAINS)
N_FACTS = len(SUFFIX) * len(FIELDS)          # 15

# 全部 series 小写形式（用于「无线索」断言）
ALL_SERIES_TOKENS = tuple(s.lower() for s in SERIES)


# ══════════════════════════════════════════════════════════════════
# 训练 / 探针
# ══════════════════════════════════════════════════════════════════
def train_pairs(domain: str):
    """规范问法（域内 2 条/事实）→ 训练对。全部含完整线索。"""
    out = []
    for e, fields in DOMAINS[domain]["entities"].items():
        for f, v in fields.items():
            out.append((f"{e} 的{f}是多少？", v))
            out.append((f"请问 {e} 的{f}是？", v))
    return out


# 每档的**留出改写**问法（训练中未出现）。`{e}`=实体名, `{f}`=字段, `{sfx}`=后缀
#
# ★ 无线索的措辞被**显式命名**，因为 G3-9 的 S5 要用它们训练「弃权类」：
#   其中 `T_c` **只用于训练、从不作为测试档** ⇒ 可测「弃权能力能否迁移到
#   **没见过的**无线索措辞」（这是「训练弃权类」这条路线是否可用的关键）。
CUELESS_TMPL = {
    "T_a": "该航天器的{f}是多少？",        # ← 同时也是 `generic` 测试档
    "T_b": "{sfx} 号的{f}是多少？",         # ← 同时也是 `partial` 测试档
    "T_c": "关于这套系统，{f}是多少？",     # ★ 仅用于训练弃权类，从不作测试档
}

STYLE_TMPL = {
    # 完整线索（阳性对照：必须高）
    "cue":     "关于 {e}，它的{f}是多少？",
    # 部分线索：丢掉 series，只留后缀 ⇒ 在 N 个域间歧义
    "partial": CUELESS_TMPL["T_b"],
    # 无线索：通用指代，prompt 内不含任何域信息
    "generic": CUELESS_TMPL["T_a"],
}
# `coref`：线索在**上一轮**，当前轮无线索
COREF_T1 = "{e} 是什么？"
COREF_T2 = "那它的{f}是多少？"


def cueless_prompts(which=("T_c",)):
    """给定无线索措辞集合 → prompt 列表（用于训练/评测「弃权类」）。

    ★ 与探针**逐字一致**（selfcheck 会断言 T_a/T_b 与 generic/partial 档完全相同），
      否则「训练弃权类」的收益会被措辞差异污染。
    """
    out = set()
    for t in which:
        tmpl = CUELESS_TMPL[t]
        for f in FIELDS:
            if "{sfx}" in tmpl:
                for sfx in SUFFIX:
                    out.add(tmpl.format(f=f, sfx=sfx))
            else:
                out.add(tmpl.format(f=f))
    return sorted(out)


def eval_items(domain: str, style: str = "cue"):
    """留出探针。`style ∈ {cue, partial, generic}`。"""
    out = []
    for e, fields in DOMAINS[domain]["entities"].items():
        sfx = e.split("-")[-1]
        for f, v in fields.items():
            q = STYLE_TMPL[style].format(e=e, f=f, sfx=sfx)
            out.append({"id": f"{domain}::{style}::{e}::{f}",
                        "kind": "acquire", "prompt": q, "answer": v,
                        "domain": domain, "style": style})
    return out


def coref_items(domain: str):
    """`coref` 探针：`(turn1, turn2)`，线索只在 turn1。"""
    out = []
    for e, fields in DOMAINS[domain]["entities"].items():
        for f, v in fields.items():
            out.append({"id": f"{domain}::coref::{e}::{f}",
                        "kind": "acquire", "prefix": COREF_T1.format(e=e),
                        "prompt": COREF_T2.format(f=f), "answer": v,
                        "domain": domain, "style": "coref"})
    return out


def n_facts(domain: str):
    return sum(len(f) for f in DOMAINS[domain]["entities"].values())


# ══════════════════════════════════════════════════════════════════
# 自检：★ 这些断言就是本实验的**单轴控制**与**操纵检查**
# ══════════════════════════════════════════════════════════════════
def selfcheck(verbose=True):
    ok = []
    # (1) 每域 15 条事实，事实键唯一
    for d in DOM_NAMES:
        keys = [(e, f) for e, fs in DOMAINS[d]["entities"].items() for f in fs]
        assert len(keys) == len(set(keys)) == N_FACTS, f"{d} 事实键异常"
    ok.append(f"每域 {N_FACTS} 条事实、键唯一")

    # (2) ★ 答案全局唯一 ⇒ 错路由必产出可检出错误
    allv = [v for d in DOM_NAMES for e in DOMAINS[d]["entities"].values()
            for v in e.values()]
    assert len(allv) == len(set(allv)), "答案不全局唯一 ⇒ 错路由可能被掩盖"
    ok.append(f"答案全局唯一（{len(allv)} 条）")

    # (3) ★ 字段名跨域全同 ⇒ 字段不携带域信息
    fs0 = set(next(iter(DOMAINS[DOM_NAMES[0]]["entities"].values())))
    for d in DOM_NAMES:
        assert set(next(iter(DOMAINS[d]["entities"].values()))) == fs0
    ok.append("字段名跨域全同（字段无域信息）")

    # (4) ★ 实体后缀跨域全同 ⇒ `partial` 真的歧义
    sfx = [frozenset(e.split("-")[-1] for e in DOMAINS[d]["entities"])
           for d in DOM_NAMES]
    assert len(set(sfx)) == 1, "实体后缀不跨域共享 ⇒ partial 档不歧义"
    assert sfx[0] == frozenset(str(s) for s in SUFFIX)
    ok.append("实体后缀跨域全同（partial 档歧义）")

    # (5) ★ 操纵检查：`generic`/`partial` 探针**必须不含**任何 series 词
    for style in ("partial", "generic"):
        for d in DOM_NAMES:
            for it in eval_items(d, style):
                low = it["prompt"].lower()
                hit = [s for s in ALL_SERIES_TOKENS if s in low]
                assert not hit, f"{style} 探针泄露线索 {hit}: {it['prompt']!r}"
    ok.append("partial/generic 探针不含任何 series 线索（操纵检查通过）")

    # (6) ★ `cue` 探针必须**恰好**含本域 series、且不含他域 series
    for d in DOM_NAMES:
        own = DOMAINS[d]["series"].lower()
        for it in eval_items(d, "cue"):
            low = it["prompt"].lower()
            assert own in low, f"cue 探针缺本域线索: {it['prompt']!r}"
            bad = [s for s in ALL_SERIES_TOKENS if s != own and s in low]
            assert not bad, f"cue 探针含他域线索 {bad}: {it['prompt']!r}"
    ok.append("cue 探针恰含本域线索、不含他域线索")

    # (7) ★ `coref`：turn2 无线索、turn1 恰含本域线索（上下文是唯一信息源）
    for d in DOM_NAMES:
        own = DOMAINS[d]["series"].lower()
        for it in coref_items(d):
            low = it["prompt"].lower()
            bad = [s for s in ALL_SERIES_TOKENS if s in low]
            assert not bad, f"coref turn2 泄露线索 {bad}: {it['prompt']!r}"
            assert own in it["prefix"].lower(), "coref turn1 缺本域线索"
    ok.append("coref：turn2 无线索 / turn1 含本域线索")

    # (8) series 之间不能互为子串（否则 (6) 的存在性检查会误判）
    for a in ALL_SERIES_TOKENS:
        for b in ALL_SERIES_TOKENS:
            if a != b:
                assert a not in b, f"series 互为子串: {a} ⊂ {b}"
    ok.append("series 两两不为子串")

    # (9) ★ 无线索措辞与测试档**逐字一致**（否则 S5「训练弃权类」的收益不可归因）
    g = {it["prompt"] for d in DOM_NAMES for it in eval_items(d, "generic")}
    p = {it["prompt"] for d in DOM_NAMES for it in eval_items(d, "partial")}
    assert g == set(cueless_prompts(["T_a"])), "T_a 措辞 ≠ generic 档"
    assert p == set(cueless_prompts(["T_b"])), "T_b 措辞 ≠ partial 档"
    # T_c 必须与两档**都不重叠**（它只用于训练）
    tc = set(cueless_prompts(["T_c"]))
    assert not (tc & g) and not (tc & p), "T_c 与测试档重叠 ⇒ 迁移检验失效"
    ok.append("T_a/T_b 与 generic/partial 档逐字一致；T_c 与两档不重叠")
    return ok


if __name__ == "__main__":
    passed = selfcheck()
    print(f"[topic_domains] 域数={len(DOM_NAMES)}  每域 {N_FACTS} 条事实")
    for d in DOM_NAMES[:3]:
        it = eval_items(d, "cue")[0]
        pa = eval_items(d, "partial")[0]
        ge = eval_items(d, "generic")[0]
        cf = coref_items(d)[0]
        print(f"  {d:<11} cue    : {it['prompt']} → {it['answer']}")
        print(f"  {'':<11} partial: {pa['prompt']} → {pa['answer']}")
        print(f"  {'':<11} generic: {ge['prompt']} → {ge['answer']}")
        print(f"  {'':<11} coref  : [{cf['prefix']}] [{cf['prompt']}] → {cf['answer']}")
    # 同类探针在**不同域**上必须问法相同（否则档间比较混入措辞差异）
    for style in ("partial", "generic"):
        s0 = [it["prompt"] for it in eval_items(DOM_NAMES[0], style)]
        for d in DOM_NAMES[1:]:
            assert [it["prompt"] for it in eval_items(d, style)] == s0, \
                f"{style} 档问法跨域不一致"
    print("[topic_domains] partial/generic 档问法跨域完全一致 ✅（同一 prompt，"
          "不同域 ⇒ 只能靠弃权或猜）")
    print("[topic_domains] ✅ 自检通过：" + "；".join(passed))
