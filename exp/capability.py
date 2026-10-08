"""能力基准（目标3 前置）：**可验证**的留出基准，用于测「核心能力是否退化」。

为什么不能用现有语料 CE：目标3 的「漂移预算 / 核心能力未退化」需要一个**有真值**的量。
本项目铁律是「A 级真值必须落在可复算的地方」⇒ 本基准的四类子任务**全部程序化判分**：

  1. `math`   —— 四则运算，答案由 Python 算出，**精确匹配**
  2. `format` —— 字符串变换/计数，答案由 Python 算出，**精确匹配**
  3. `code`   —— 生成小函数，**子进程里跑单测**（这等价于 A 级真值）
  4. `lm`     —— 留出语料的 **token 准确率 + CE**（教师强制，不生成）

设计约束：
  * 生成必须 `enable_thinking=False`（否则长思考链截断 ⇒ 判成「答错」，见 SLATIS 教训）。
  * 判分要**宽容到只看答案**（允许前后缀、标点、```` ``` ```` 围栏），否则测的是格式而非能力。
  * 每类都报**基线**精度；退化 = 基线 − 之后。
"""
from __future__ import annotations

import os
import re
import json
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from s5_edit import render, generate  # noqa: E402
from common import encode_batch, load_calib_texts, CALIB_PATH  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SEED = 20261005


# ══════════════════════════════════════════════════════════════════════
# 1) math / format：答案由 Python 算出
# ══════════════════════════════════════════════════════════════════════

def build_math_items():
    import random
    rng = random.Random(SEED)
    items = []
    for i in range(24):
        kind = i % 4
        if kind == 0:
            a, b = rng.randint(23, 99), rng.randint(13, 89)
            q, ans = f"计算 {a} × {b}，只输出结果数字。", str(a * b)
        elif kind == 1:
            a, b = rng.randint(101, 899), rng.randint(101, 799)
            q, ans = f"计算 {a} + {b}，只输出结果数字。", str(a + b)
        elif kind == 2:
            a, b = rng.randint(400, 999), rng.randint(100, 380)
            q, ans = f"计算 {a} - {b}，只输出结果数字。", str(a - b)
        else:
            b = rng.randint(3, 19)
            ans_v = rng.randint(11, 60)
            q, ans = (f"计算 {b * ans_v} ÷ {b}，只输出结果数字。", str(ans_v))
        items.append({"id": f"math{i:02d}", "kind": "math",
                      "prompt": q, "answer": ans, "check": "exact"})
    return items


WORDS = ["strawberry", "mississippi", "abracadabra", "examination",
         "parallel", "committee", "environment", "rhythm"]


def build_format_items():
    items = []
    for i, w in enumerate(WORDS):
        items.append({"id": f"fmt{i:02d}a", "kind": "format",
                      "prompt": f"把单词 {w} 的字母按倒序输出，只输出结果。",
                      "answer": w[::-1], "check": "exact"})
        c = "s" if w.count("s") else max(set(w), key=w.count)
        items.append({"id": f"fmt{i:02d}b", "kind": "format",
                      "prompt": f"统计单词 {w} 中字母 {c} 出现的次数，只输出数字。",
                      "answer": str(w.count(c)), "check": "exact"})
        items.append({"id": f"fmt{i:02d}c", "kind": "format",
                      "prompt": f"把单词 {w} 转成大写，只输出结果。",
                      "answer": w.upper(), "check": "exact"})
    return items


# ══════════════════════════════════════════════════════════════════════
# 2) code：子进程跑单测（A 级真值）
# ══════════════════════════════════════════════════════════════════════

CODE_TASKS = [
    ("写一个 Python 函数 `add(a, b)` 返回两数之和。只输出代码，不要解释。",
     "def add(a, b): return a + b", {"add(2,3)==5", "add(-1,1)==0"}),
    ("写一个 Python 函数 `is_even(n)` 返回 n 是否为偶数。只输出代码。",
     "def is_even(n): return n % 2 == 0", {"is_even(4)==True", "is_even(7)==False"}),
    ("写一个 Python 函数 `maximum(xs)` 返回列表最大值。只输出代码。",
     "def maximum(xs): return max(xs)", {"maximum([1,5,3])==5", "maximum([-2,-9])==-2"}),
    ("写一个 Python 函数 `reverse_str(s)` 返回反转字符串。只输出代码。",
     "def reverse_str(s): return s[::-1]", {"reverse_str('abc')=='cba'"}),
    ("写一个 Python 函数 `count_char(s, c)` 返回 c 在 s 中出现次数。只输出代码。",
     "def count_char(s, c): return s.count(c)", {"count_char('miss','s')==2"}),
    ("写一个 Python 函数 `factorial(n)` 返回 n 的阶乘（n>=0）。只输出代码。",
     "def factorial(n):\n    r = 1\n    for i in range(2, n+1): r *= i\n    return r",
     {"factorial(0)==1", "factorial(5)==120"}),
    ("写一个 Python 函数 `sum_list(xs)` 返回列表元素之和。只输出代码。",
     "def sum_list(xs): return sum(xs)", {"sum_list([1,2,3])==6", "sum_list([])==0"}),
    ("写一个 Python 函数 `is_palindrome(s)` 判断字符串是否回文。只输出代码。",
     "def is_palindrome(s): return s == s[::-1]",
     {"is_palindrome('aba')==True", "is_palindrome('ab')==False"}),
]


def build_code_items():
    return [{"id": f"code{i:02d}", "kind": "code", "prompt": q,
             "answer": ref, "tests": list(ts), "check": "unittest"}
            for i, (q, ref, ts) in enumerate(CODE_TASKS)]


def extract_code(text):
    """从生成里抠出代码：优先围栏，否则整段。"""
    m = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, re.S)
    if m:
        return "\n".join(m)
    return text


def run_unittest(code, tests, timeout=10):
    """在子进程里 exec 代码并断言 tests；返回 (ok, detail)。"""
    # ★ 坑：不要把测试表达式嵌进 assert 的 message 里 —— 表达式含单引号时会
    #   提前终止字符串字面量 ⇒ SyntaxError（本 harness 自己踩过）。用序号做 message。
    body = code + "\n" + "\n".join(
        f"assert {t}, 'test #{i} failed'" for i, t in enumerate(tests)) \
        + "\nprint('OK')\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(body)
            path = f.name
        p = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        os.unlink(path)
        return (p.returncode == 0 and "OK" in p.stdout,
                (p.stderr or p.stdout)[-300:])
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


# ══════════════════════════════════════════════════════════════════════
# 3) 判分
# ══════════════════════════════════════════════════════════════════════

def normalize(s):
    s = s.strip()
    s = re.sub(r"^[`*_\s]+|[`*_\s]+$", "", s)
    s = s.replace("**", "").replace("：", ":")
    return s.strip()


def grade_exact(resp, answer):
    r = normalize(resp)
    if r.lower() == answer.lower():
        return True
    # 只取第一行 / 第一个 token 段再比一次（避免被多余解释拖累）
    first = normalize(r.splitlines()[0]) if r else ""
    if first.lower() == answer.lower():
        return True
    # 数字题：抓第一个整数（必须独立，不能被更长数字包含）
    if answer.lstrip("-").isdigit():
        m = re.search(r"-?\d+", r)
        if m and m.group(0) == answer:
            return True
        return False
    # ★ 非数字题：允许答案出现在一段话里（答案本身足够特异，仍可验证）。
    #   否则测的是“会不会只输出答案”这种格式能力，而不是能力本身。
    return re.search(rf"(?<![0-9A-Za-z]){re.escape(answer)}(?![0-9A-Za-z])",
                     r, re.I) is not None


@torch.no_grad()
def eval_generation(model, tokenizer, items, max_new_tokens=40, log=False):
    per = []
    for it in items:
        try:
            txt = generate(model, tokenizer, it["prompt"],
                           max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            txt = ""
            if log:
                print(f"    [warn] 生成失败 {it['id']}: {exc}")
        if it["check"] == "unittest":
            ok, detail = run_unittest(extract_code(txt), it["tests"])
        else:
            ok, detail = grade_exact(txt, it["answer"]), ""
        per.append({"id": it["id"], "kind": it["kind"], "ok": bool(ok),
                    "prompt": it["prompt"][:60], "resp": txt[:160],
                    "detail": detail})
    return per


@torch.no_grad()
def eval_lm(model, tokenizer, texts, seq_len=256, batch=4):
    tot_ce, tot_n, correct, ntok = 0.0, 0, 0, 0
    for i in range(0, len(texts), batch):
        enc = encode_batch(tokenizer, texts[i:i + batch], seq_len)
        logits = model(**enc).logits
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:]
        loss = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tot_ce += float((loss * mask).sum())
        pred = lp.argmax(-1)
        correct += int(((pred == tgt) & mask.bool()).sum())
        ntok += int(mask.sum())
        tot_n += 1
    return {"ce": tot_ce / max(ntok, 1), "token_acc": correct / max(ntok, 1),
            "n_tokens": ntok}


def summarize(per):
    out = {}
    for k in ("math", "format", "code"):
        v = [p for p in per if p["kind"] == k]
        out[k] = {"acc": (sum(p["ok"] for p in v) / len(v)) if v else float("nan"),
                  "n": len(v), "n_ok": sum(p["ok"] for p in v)}
    v = [p for p in per if p["kind"] in ("math", "format")]
    out["verifiable"] = {"acc": (sum(p["ok"] for p in v) / len(v)) if v else float("nan"),
                         "n": len(v), "n_ok": sum(p["ok"] for p in v)}
    return out


def full_battery(model, tokenizer, lm_texts, log=False):
    items = build_math_items() + build_format_items() + build_code_items()
    per = eval_generation(model, tokenizer, items, log=log)
    res = summarize(per)
    res["lm"] = eval_lm(model, tokenizer, lm_texts)
    res["_per_item"] = per
    return res


def lm_texts_heldout(n=32, start=2000):
    return load_calib_texts(CALIB_PATH, n, start=start)


if __name__ == "__main__":
    # 自检：不加载模型也能验分判分器与代码执行器
    mi = build_math_items() + build_format_items()
    bad = 0
    for it in mi:
        pass
    print(f"[cap] math={len(build_math_items())} format={len(build_format_items())} "
          f"code={len(build_code_items())}")
    # 判分器正/负例
    assert grade_exact("2961", "2961")
    assert grade_exact("结果是 2961。", "2961")
    assert grade_exact("```\nyrrebwarts\n```", "yrrebwarts")
    assert not grade_exact("2962", "2961")
    print("[cap] 判分器正/负例 ✅")
    # 代码执行器：参考实现应通过，坏实现应失败
    for q, ref, ts in CODE_TASKS:
        ok, _ = run_unittest(ref, ts)
        assert ok, f"参考实现未通过：{q}"
        ok2, _ = run_unittest("def add(a,b): return a-b", ["add(2,3)==5"]) \
            if "add" in q else (False, "")
        assert not ok2
    print(f"[cap] {len(CODE_TASKS)} 个参考实现全部通过单测，且错误实现被拒 ✅")
