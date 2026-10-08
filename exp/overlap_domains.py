"""三对**重叠程度递增**的域，用来检验门控的适用边界（`g3_8_overlap.py` 用）。

★ 设计原则：三对之间**只改变一个变量** —— 「判别性 token 是什么」。
  否则门控一旦失败，无法指出是哪个性质导致的。

  每对含 2 个域，各 3 实体 × 5 字段 = 15 事实，规范化问法 2 条 → 30 训练对，
  留出改写问法 1 条 → 15 探针（问法在训练中未出现）。

  ┌ P1 `alpha` vs `beta` —— **实体名不同、字段名全同**
  │    判别信号 = 独一无二的实体名（= 原实验的设定，作参考基准）
  │    预期：容易，门控应接近 100%
  ├ P2 `gamma` vs `delta` —— **实体名完全相同、字段名不同**
  │    判别信号 = 字段名 token。实体名（在平均池化里占主导）**共享**
  │    ⇒ 对 mean-pool 是真考验
  └ P3 `eps` vs `zeta` —— **实体名相同、字段名相同、只差一个限定词**
       问法只差「北极」/「南极」两个 token，其余完全一致
       ⇒ 判别信号的**token 占比最小**（约 2/14），最难

★ 为什么 P3 是公平的（不是无解的）：两个限定词是不同的实词，在残差流里可区分；
  问题只是「平均池化把它稀释到什么程度」、以及「换一种池化能否救回来」。
  ⇒ 所以 `g3_8` 会跑 (层 × 池化) 的**扫描**，而不只报一个数：
     若 mean 失败而 last/max 成功 ⇒ 结论是「池化选择问题」，路线仍可行；
     若全部失败 ⇒ 结论强得多：这类域**无法**用简单读出路由。
"""
from __future__ import annotations

PAIR_LIST = ["P1_disjoint_entity", "P2_shared_entity", "P3_single_token"]

PAIRS = {
    "P1_disjoint_entity": {
        "a": "alpha", "b": "beta",
        "why": ("实体名不同、字段名全同 ⇒ 判别信号是独一无二的实体名"
                "（原实验设定，参考基准）"),
    },
    "P2_shared_entity": {
        "a": "gamma", "b": "delta",
        "why": ("实体名**完全相同**、字段名不同 ⇒ 判别信号只剩字段名；"
                "实体名在平均池化里占主导且共享"),
    },
    "P3_single_token": {
        "a": "eps", "b": "zeta",
        "why": ("实体名相同、字段名相同，问法只差「北极」/「南极」两个 token"
                "⇒ 判别信号 token 占比最小"),
    },
}

_F = ["轨道高度", "载荷质量", "发射年份", "在轨寿命", "功率"]
_F2 = ["质保年限", "单位成本", "交付周期", "发射服务商", "保险额度"]
_E3 = ["Nimbus-7", "Zephyr-4", "Solstice-2"]

DOMAINS = {
    # ── P1：不同实体、同字段 ────────────────────────────────────────
    "alpha": {"pair": "P1_disjoint_entity", "side": "a",
              "title": "Nimbus 系列（甲）",
              "entities": {
                  "Nimbus-7":  dict(zip(_F, ["612 千米", "3400 千克", "2031",
                                             "11 年", "4.2 千瓦"])),
                  "Zephyr-4":  dict(zip(_F, ["780 千米", "1900 千克", "2029",
                                             "7 年", "2.6 千瓦"])),
                  "Solstice-2": dict(zip(_F, ["1200 千米", "5200 千克", "2033",
                                              "15 年", "7.8 千瓦"])),
              }},
    "beta": {"pair": "P1_disjoint_entity", "side": "b",
             "title": "Larkspur 系列（乙）",
             "entities": {
                 "Larkspur-5": dict(zip(_F, ["450 千米", "880 千克", "2028",
                                             "5 年", "1.1 千瓦"])),
                 "Quillon-8":  dict(zip(_F, ["980 千米", "2600 千克", "2030",
                                             "9 年", "3.3 千瓦"])),
                 "Mistral-3":  dict(zip(_F, ["1600 千米", "6100 千克", "2035",
                                             "18 年", "9.5 千瓦"])),
             }},
    # ── P2：同实体、不同字段 ────────────────────────────────────────
    "gamma": {"pair": "P2_shared_entity", "side": "a",
              "title": "Nimbus 系列技术规格",
              "entities": {
                  "Nimbus-7":  dict(zip(_F, ["612 千米", "3400 千克", "2031",
                                             "11 年", "4.2 千瓦"])),
                  "Zephyr-4":  dict(zip(_F, ["780 千米", "1900 千克", "2029",
                                             "7 年", "2.6 千瓦"])),
                  "Solstice-2": dict(zip(_F, ["1200 千米", "5200 千克", "2033",
                                              "15 年", "7.8 千瓦"])),
              }},
    "delta": {"pair": "P2_shared_entity", "side": "b",
              "title": "Nimbus 系列商务条款",
              "entities": {
                  "Nimbus-7":  dict(zip(_F2, ["36 个月", "2.4 亿美元", "22 周",
                                              "苍穹发射", "8.5 亿美元"])),
                  "Zephyr-4":  dict(zip(_F2, ["24 个月", "1.1 亿美元", "14 周",
                                              "海角航天", "3.9 亿美元"])),
                  "Solstice-2": dict(zip(_F2, ["48 个月", "3.8 亿美元", "30 周",
                                               "北星运载", "12.6 亿美元"])),
              }},
    # ── P3：同实体、同字段头、只差限定词 ────────────────────────────
    # ★ 用「限定词 + 字段」组合成 5 个字段 ⇒ 与 P1/P2 一样 15 条事实，
    #   否则每个域只有 3 条训练问法，门控根本没法学（这是第一版的错误）。
    "eps": {"pair": "P3_single_token", "side": "a",
            "title": "Nimbus 系列北极轨道",
            "entities": {e: {f"在北极轨道上的{f}": v for f, v in
                             zip(_F, vals)}
                         for e, vals in zip(_E3, [
                             ["640 千米", "3250 千克", "2031", "11 年", "4.2 千瓦"],
                             ["815 千米", "1820 千克", "2029", "7 年", "2.6 千瓦"],
                             ["1260 千米", "5050 千克", "2033", "15 年", "7.8 千瓦"],
                         ])}},
    "zeta": {"pair": "P3_single_token", "side": "b",
             "title": "Nimbus 系列南极轨道",
             "entities": {e: {f"在南极轨道上的{f}": v for f, v in
                              zip(_F, vals)}
                          for e, vals in zip(_E3, [
                              ["470 千米", "920 千克", "2028", "4 年", "1.3 千瓦"],
                              ["1010 千米", "2740 千克", "2030", "13 年", "3.6 千瓦"],
                              ["1680 千米", "6480 千克", "2035", "6 年", "9.1 千瓦"],
                          ])}},
}

_ALT_Q = ("请问 {e} 的{f}是？", "{f}——{e} 是多少？")
# ★ 同时给出**更远的**改写（P1/P2/P3 共用），用于检验门控是否只在模板上有效
_FAR_Q = ("关于 {e}，它的{f}是多少？",)


def train_pairs(domain: str):
    """规范问法（域内 2 条/事实）→ 训练对。"""
    out = []
    for e, fields in DOMAINS[domain]["entities"].items():
        for f, v in fields.items():
            out.append((f"{e} 的{f}是什么？", v))
            out.append((f"{e} 的{f}？", v))
    return out


def eval_items(domain: str, style: str = "near"):
    """留出改写问法 → 习得/路由探针。`style='near'` 近改写，`'far'` 远改写。"""
    q = _ALT_Q[0] if style == "near" else _FAR_Q[0]
    out = []
    for e, fields in DOMAINS[domain]["entities"].items():
        for f, v in fields.items():
            out.append({"id": f"{domain}::{e}::{f}", "prompt": q.format(e=e, f=f),
                        "answer": v, "domain": domain, "style": style})
    return out


def n_facts(domain: str):
    return sum(len(f) for f in DOMAINS[domain]["entities"].values())


if __name__ == "__main__":
    for pname in PAIR_LIST:
        p = PAIRS[pname]
        print(f"\n=== {pname}: {p['a']} vs {p['b']} ===")
        print(f"    {p['why']}")
        for side in ("a", "b"):
            d = p[side]
            tr, ev = train_pairs(d), eval_items(d)
            print(f"  {d:<8} 实体={list(DOMAINS[d]['entities'])}")
            print(f"           训练例: {tr[0][0]} → {tr[0][1]}")
            print(f"           留出例: {ev[0]['prompt']} → {ev[0]['answer']}")
    # ── 自检：P1 字段名相同；P2 实体名相同；P3 只差限定词 ─────────────
    print("\n[overlap] 难度轴自检")
    for pname in PAIR_LIST:
        p = PAIRS[pname]
        ea = set(DOMAINS[p["a"]]["entities"])
        eb = set(DOMAINS[p["b"]]["entities"])
        fa = {f for d in DOMAINS[p["a"]]["entities"].values() for f in d}
        fb = {f for d in DOMAINS[p["b"]]["entities"].values() for f in d}
        print(f"  {pname:<20} 实体名交集={len(ea & eb)}/{len(ea)}  "
              f"字段名交集={len(fa & fb)}/{len(fa)}")
    # 判别性 token 占比（越少越难）
    for pname in PAIR_LIST:
        p = PAIRS[pname]
        qa = train_pairs(p["a"])[0][0]
        qb = train_pairs(p["b"])[0][0]
        sa, sb = set(qa), set(qb)
        print(f"  {pname:<20} 首条问法字符集差异={len(sa ^ sb)}  例: {qa!r} vs {qb!r}")
    # 三对中任一域内 实体名×字段 必须唯一（否则事实撞车）
    for pname in PAIR_LIST:
        p = PAIRS[pname]
        for side in ("a", "b"):
            d = p[side]
            keys = [(e, f) for e, fs in DOMAINS[d]["entities"].items()
                    for f in fs]
            assert len(keys) == len(set(keys)), f"{d} 事实键重复"
            assert n_facts(d) == 15, f"{d} 事实数 {n_facts(d)} ≠ 15"
    # P3 的两个域答案必须完全不同（否则错路由无法察觉）
    va = {v for _, v in train_pairs("eps")}
    vb = {v for _, v in train_pairs("zeta")}
    assert not (va & vb), f"P3 两域答案重叠：{va & vb}"
    print("[overlap] ✅ 自检通过（事实键唯一、各 15 条、P3 答案不重叠）")
