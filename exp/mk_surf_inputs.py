"""为 surf_CV 检验导出输入（不依赖 torch / 模型）。

导出三样东西到一个 JSON：
  1. 160 条条目的结构（id / entity / attr / gold / design_known）
     —— design_known 是我的**设计先验**（gold 非空），不是实测边界；
  2. 实测边界的 **test 半边** 标签（来自 g3_15 的逐条落盘）：
     known_test 26 条 + unknown_test 25 条 = 51 条；
  3. 病态条目（g3_16 的加固口径剔除的两条）。

★ 为什么只有 test 半边有实测标签：
  `known_train`(27) / `unknown_train`(25) 的 id 从未落盘，
  任何 JSON 里都只有计数（g3_13 / g3_14 / g3_15 均如此）。
  要拿到完整 103 条，只能在实验机上重跑 P0×3 分类后导出。
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import real_kb_data as KB  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def main() -> None:
    items = KB.all_items()

    s15 = json.loads((OUT / "g3_15_known_slice_items.json").read_text("utf-8"))
    s16 = json.loads((OUT / "g3_16_known_slice_hardened.json").read_text("utf-8"))

    known_test = list(s15["known_test_ids"])
    unknown_test = [it["id"] for it in s15["items"]["base_unknown_test"]]

    pathological = []
    for cand in ("蒙娜丽莎::画中人物的身份", "哈姆雷特::丹麦王子"):
        if cand in known_test:
            pathological.append(cand)

    payload = {
        "note": "design_known 是设计先验，不是实测边界；measured_test 才是实测标签",
        "n_items": len(items),
        "items": items,
        "measured_known_test": known_test,
        "measured_unknown_test": unknown_test,
        "pathological": pathological,
        "counts": {
            "design_known": sum(1 for i in items if i["known_hint"]),
            "design_obscure": sum(1 for i in items if not i["known_hint"]),
            "measured_known_test": len(known_test),
            "measured_unknown_test": len(unknown_test),
        },
    }

    OUT.mkdir(exist_ok=True)
    dst = OUT / "surf_cv_inputs.json"
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(f"[surf] wrote {dst}")
    print(f"[surf] items={payload['n_items']}  "
          f"design known/obscure={payload['counts']['design_known']}/"
          f"{payload['counts']['design_obscure']}")
    print(f"[surf] measured test known/unknown="
          f"{len(known_test)}/{len(unknown_test)}  "
          f"pathological={pathological}")


if __name__ == "__main__":
    main()
