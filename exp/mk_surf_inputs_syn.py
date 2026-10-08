"""为 surf_CV 导出**合成切片**（`abstain_data.py`）的输入——不需要模型。

合成切片的标签是**构造定义**的（supported → 答案，unsupported → 弃权），
所以可以完全离线枚举。

★ 这里要回答的对照问题：
  真实边界切片在**评测轴**（未见 (实体,属性) 对）上 surf_CV = 0.82 ⇒ 没过；
  合成切片在**评测轴**（未见 series）上应当 ≈ 0.5 ⇒ 过。

  关键：能用于测试的表层规则只能建立在**训练与测试共享的轴**上。
  · 合成：训练 series ⊂ SUPPORTED ∪ UNK_MULTI，测试 series = UNK_HOLD
          ⇒ series 轴在测试时**不可用**；可用的只有 字段 / 后缀 / 模板
  · 真实：测试条目与训练**共享实体**，新的是 (实体,属性) 对
          ⇒ 实体轴可用（但两类都有），属性名风格轴可用且**泄漏**
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import abstain_data as AD  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def main() -> None:
    rows = []

    def add(series, tmpl, label, role):
        for e in AD._entities(series):
            for f in AD.FIELDS:
                rows.append({
                    "id": f"{e}::{f}::{tmpl}",
                    "session": role,
                    "series": series,
                    "suffix": e.split("-")[1],
                    "field": f,
                    "tmpl": tmpl,
                    "label": label,          # 1 = abstain, 0 = answer
                })

    # 训练集：与 `sft_cross_3unk` 一致 —— 答案与弃权**共用 F1/F2**（交叉模板）
    for s in AD.SUPPORTED:
        for t in ("F1", "F2"):
            add(s, t, 0, "train")
    for s in AD.UNK_MULTI:
        for t in ("F1", "F2"):
            add(s, t, 1, "train")

    # 评测集：未见 series（UNK_HOLD）× 未见模板 F4；两侧都要
    add(AD.UNK_HOLD, "F4", 1, "test")
    for s in AD.SUPPORTED:
        add(s, "F4", 0, "test")

    tr = [r for r in rows if r["session"] == "train"]
    te = [r for r in rows if r["session"] == "test"]

    payload = {
        "note": "synthetic slice; labels are by construction. "
                "train = crossed templates (F1/F2 for both classes); "
                "test = unseen series UNK_HOLD with unseen template F4",
        "fields": AD.FIELDS,
        "supported": AD.SUPPORTED,
        "unk_multi": AD.UNK_MULTI,
        "unk_hold": AD.UNK_HOLD,
        "items": rows,
        "n_train": len(tr),
        "n_test": len(te),
        "train_known": sum(1 for r in tr if r["label"] == 0),
        "train_unknown": sum(1 for r in tr if r["label"] == 1),
    }

    OUT.mkdir(exist_ok=True)
    dst = OUT / "surf_cv_inputs_syn.json"
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(f"[syn] wrote {dst}")
    print(f"[syn] train={len(tr)} (answer {payload['train_known']} / "
          f"abstain {payload['train_unknown']})  test={len(te)}")
    print(f"[syn] fields={AD.FIELDS}")
    print(f"[syn] SUPPORTED={AD.SUPPORTED}")
    print(f"[syn] UNK_MULTI={AD.UNK_MULTI}  UNK_HOLD={AD.UNK_HOLD}")


if __name__ == "__main__":
    main()
