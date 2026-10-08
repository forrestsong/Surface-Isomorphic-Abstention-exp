"""三个互不相同的**虚构域**（用于热更新的多域实验）。

为什么需要多个域：目标3 §8 风险三问的是「多次迭代后累积漂移」。
上一轮前提检验用的是**同一个域跑两遍** ⇒ 结构上**无法**测「域间干扰」（第二次训练
见到的还是同一批事实，只会更熟）。要测「热更新会不会覆盖上一次更新」，必须有
**互不重叠**的域。

每个域 = 3 个实体 × 5 个字段 = **15 条事实**；规范问法 2 条 → 30 条训练对；
**留出改写问法 1 条** → 15 条习得探针（问法在训练中未出现）。
域与域之间**实体名、字段名、答案词表全部不相交**，便于分离干扰。
"""
from __future__ import annotations

DOMAINS = {
    "northwind": {
        "title": "Northwind Dynamics 产品线",
        "entities": {
            "Aurora-9":  {"发布时间": "2026", "最大并发": "4096",
                          "延迟": "12 毫秒", "价格": "1900 美元", "SDK": "AuroraSDK"},
            "Borealis-3": {"发布时间": "2025", "最大并发": "1024",
                           "延迟": "45 毫秒", "价格": "700 美元", "SDK": "BorealisKit"},
            "Cascade-7": {"发布时间": "2028", "最大并发": "65536",
                          "延迟": "3 毫秒", "价格": "9800 美元", "SDK": "CascadeKit"},
        },
    },
    "halcyon": {
        "title": "Halcyon 射电阵望远镜",
        "entities": {
            "Halcyon-1": {"启用年份": "2031", "主镜口径": "8.4 米",
                          "观测波段": "近红外", "站点海拔": "4200 米", "首席工程师": "韦岚"},
            "Halcyon-2": {"启用年份": "2034", "主镜口径": "12.6 米",
                          "观测波段": "亚毫米波", "站点海拔": "5100 米", "首席工程师": "郭澈"},
            "Halcyon-3": {"启用年份": "2037", "主镜口径": "25.0 米",
                          "观测波段": "米波", "站点海拔": "2800 米", "首席工程师": "祁霜"},
        },
    },
    "vela": {
        "title": "Vela 编程语言版本",
        "entities": {
            "Vela 1.0": {"发布年份": "2024", "类型系统": "渐进类型",
                         "并发模型": "Actor", "包管理器": "Velp", "设计者": "苏格"},
            "Vela 2.0": {"发布年份": "2027", "类型系统": "依赖类型",
                         "并发模型": "CSP", "包管理器": "Velp2", "设计者": "闻舟"},
            "Vela 3.0": {"发布年份": "2030", "类型系统": "线性类型",
                         "并发模型": "Dataflow", "包管理器": "Velp3", "设计者": "裴野"},
        },
    },
}

_ALT_Q = ("请问 {e} 的{f}是？", "{f}——{e} 是多少？")


def train_pairs(domain: str):
    """规范问法（域内 2 条/事实）→ 训练对。"""
    d = DOMAINS[domain]
    out = []
    for e, fields in d["entities"].items():
        for f, v in fields.items():
            out.append((f"{e} 的{f}是什么？", v))
            out.append((f"{e} 的{f}？", v))
    return out


def eval_items(domain: str):
    """**留出改写问法**（训练中未出现）→ 习得探针。"""
    d = DOMAINS[domain]
    out = []
    for e, fields in d["entities"].items():
        for f, v in fields.items():
            q = _ALT_Q[0].format(e=e, f=f)
            out.append({"id": f"{domain}::{e}::{f}", "kind": "acquire",
                        "prompt": q, "answer": v})
    return out


def all_eval_items(domains=None):
    domains = domains or list(DOMAINS)
    return [it for d in domains for it in eval_items(d)]


def n_facts(domain: str):
    return sum(len(f) for f in DOMAINS[domain]["entities"].values())


if __name__ == "__main__":
    for name in DOMAINS:
        tr, ev = train_pairs(name), eval_items(name)
        print(f"{name:<10} 事实={n_facts(name):>2}  训练对={len(tr):>2}  探针={len(ev):>2}"
              f"   例: {tr[0][0]} → {tr[0][1]}  |  {ev[0]['prompt']} → {ev[0]['answer']}")
    # 域间不相交自检（实体名与答案词表都不能重叠）
    import itertools
    names = {k: set(v["entities"]) for k, v in DOMAINS.items()}
    for a, b in itertools.combinations(DOMAINS, 2):
        assert not (names[a] & names[b]), f"{a} 与 {b} 实体名重叠"
        ans_a = {v for _, v in train_pairs(a)}
        ans_b = {v for _, v in train_pairs(b)}
        assert not (ans_a & ans_b), f"{a} 与 {b} 答案值重叠：{ans_a & ans_b}"
    print("[domains] ✅ 三域实体名与答案值均不相交")
