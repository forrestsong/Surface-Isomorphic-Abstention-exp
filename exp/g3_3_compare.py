"""G3-3 汇总两个注入层带的对照，产出 out/g3_compare.json（供报告直接引用）。

对照的问题：**遗忘是否依赖「注入在哪一层」？**
  arm A = L12–17（目标2 自我认知带）
  arm B = L26–31（目标1 奇点带：top-1% 集中度与超激活峰所在）
若 B 的遗忘显著大于 A，则「目标1 的奇点图谱」对目标3 的注入点选择就是有用的。
"""
from __future__ import annotations

import os
import json
import glob


def jload(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def summarize_arm(d):
    cfg = d["config"]
    rows = {}
    for s in d["snapshots"]:
        b = s["battery"]
        rows[s["tag"]] = {
            "acquire_acc": s["acquire_acc"],
            "verifiable": s["battery_verifiable_acc"],
            "math": b["math"]["acc"], "format": b["format"]["acc"],
            "code": b["code"]["acc"],
            "lm_token_acc": b["lm"]["token_acc"], "lm_ce": b["lm"]["ce"],
            "delta_ce_pct": s.get("delta_ce_pct"),
            "forget_verifiable": s.get("forget_verifiable"),
            "identity_hit": s["identity_hit"],
            "enrich_principal": (s.get("subspace") or {}).get("enrich_principal"),
            "share_principal": (s.get("subspace") or {}).get("share_principal_mean"),
            "n_above_15pct": (s.get("subspace") or {}).get("n_above_plan_15pct"),
            "n_modules": (s.get("subspace") or {}).get("n"),
        }
    return {"layers": cfg.get("layers"), "tag": cfg.get("tag", "unknown"),
            "n_train_pairs": cfg.get("n_train_pairs"), "rows": rows}


def main():
    # 参数化前的旧文件也纳入（它是一次**独立的基座读数**，是噪声底的一部分）
    all_runs = []      # (label, arm_key, data)
    for label, path, arm in (
        ("band12_17_oldgrader", "out/g3_premise.json", "band12_17"),
        ("band12_17", "out/g3_premise_band12_17.json", "band12_17"),
        ("band26_31", "out/g3_premise_band26_31.json", "band26_31"),
    ):
        d = jload(path)
        if d:
            all_runs.append((label, arm, d))
    if not all_runs:
        raise SystemExit("没有 g3_premise_*.json，先跑 g3_1_premise.py")

    # 每臂取**最新**的那次作为正式读数（band12_17 优先用新判分器那次）
    out = {}
    for label, arm, d in reversed(all_runs):
        if arm in out:
            continue
        d.setdefault("config", {}).setdefault("tag", arm)
        d["config"]["_source"] = label
        out[arm] = summarize_arm(d)
    out = dict(reversed(list(out.items())))

    # 阳性对照灵敏度（任取一个 arm，两个 arm 的对照不同，分别报）
    for tag, a in out.items():
        ctrl = a["rows"].get("destructive_ctrl", {})
        base = a["rows"].get("base", {})
        a["positive_control_drop"] = (base.get("verifiable", 0)
                                      - ctrl.get("verifiable", 0))

    # 跨层带比较：真实注入后的遗忘
    cmp = {}
    for tag, a in out.items():
        for stage in ("after_d1", "after_d2"):
            r = a["rows"].get(stage, {})
            cmp.setdefault(stage, {})[tag] = {
                "forget_verifiable": r.get("forget_verifiable"),
                "delta_ce_pct": r.get("delta_ce_pct"),
                "acquire_acc": r.get("acquire_acc"),
                "enrich_principal": r.get("enrich_principal"),
            }

    # ── 仪器噪声底：同一**未被改动**的基座在多个独立进程里的读数 ──────────
    #   ★ 实测：进程内 base 与 after 逐位相同，但**跨进程** format 子基准会抖
    #   （0.46–0.54）⇒ 跨臂比较绝对值必须对着这个噪声底，否则会把仪器抖动
    #   当成"层带效应"（本实验差点犯这个错）。
    base_reads = []
    for label, arm, d in all_runs:
        s = summarize_arm(d)
        r = s["rows"].get("base")
        if r:
            base_reads.append({"run": label, "verifiable": r["verifiable"],
                               "math": r["math"], "format": r["format"],
                               "code": r["code"], "lm_token_acc": r["lm_token_acc"],
                               "lm_ce": r["lm_ce"]})
    if base_reads:
        vs = [b["verifiable"] for b in base_reads]
        mean = sum(vs) / len(vs)
        sd = (sum((x - mean) ** 2 for x in vs) / max(len(vs) - 1, 1)) ** 0.5
        noise = {"n_independent_processes": len(vs), "base_verifiable_values": vs,
                 "mean": mean, "sd": sd, "range": max(vs) - min(vs),
                 "format_values": [b["format"] for b in base_reads],
                 "lm_ce_values": [b["lm_ce"] for b in base_reads],
                 "lm_ce_range": max(b["lm_ce"] for b in base_reads)
                 - min(b["lm_ce"] for b in base_reads)}
    else:
        noise = None

    # ── 跨臂比较（习得 / 遗忘 / CE）────────────────────────────────
    cross = {}
    arms = list(out.keys())
    if len(arms) >= 2:
        for stage in ("after_d1", "after_d2"):
            row = {}
            for tag in arms:
                r = out[tag]["rows"].get(stage, {})
                row[tag] = {"acquire": r.get("acquire_acc"),
                            "forget": r.get("forget_verifiable"),
                            "delta_ce_pct": r.get("delta_ce_pct")}
            cross[stage] = row
        # 习得的跨臂差
        acq_gap = {tag: out[tag]["rows"].get("after_d2", {}).get("acquire_acc")
                   for tag in arms}
        cross["acquire_after_d2"] = acq_gap

    with open("out/g3_compare.json", "w", encoding="utf-8") as f:
        json.dump({"arms": out, "cross_arm": cross,
                   "instrument_noise_floor": noise}, f,
                  ensure_ascii=False, indent=2)

    def f2(x, fmt="{:.2f}", dash="—"):
        return dash if x is None else fmt.format(x)

    print(f"{'arm':<12}{'stage':<18}{'习得':>7}{'可验证':>8}{'遗忘':>8}"
          f"{'ΔCE%':>8}{'主奇异富集':>11}")
    for tag, a in out.items():
        for stage in ("base", "after_d1", "after_d2", "destructive_ctrl"):
            r = a["rows"].get(stage)
            if not r:
                continue
            enr = r["enrich_principal"]
            print(f"{tag:<12}{stage:<18}{r['acquire_acc']:>7.2f}"
                  f"{r['verifiable']:>8.3f}"
                  f"{f2(r['forget_verifiable'], '{:+.3f}'):>8}"
                  f"{f2(r['delta_ce_pct'], '{:+.2f}'):>8}"
                  f"{('—' if enr is None else f'x{enr:.2f}'):>11}")
    print("\n阳性对照灵敏度（可验证能力下降）：")
    for tag, a in out.items():
        print(f"  {tag}: {a['positive_control_drop']:+.3f}")
    print("\n[S9/G3] 写出 out/g3_compare.json")


if __name__ == "__main__":
    main()
