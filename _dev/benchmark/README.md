# 打牌能力基准（Play Benchmark）

给定**完美结构化事实**（无图、无识别噪声），评估模型在四类决策上的出牌质量。
识图质量由桌面回归测试覆盖（35/35），本基准只测「拿到事实后会不会打」。

## 场景集（24 个）

| 类别 | 数量 | 期望答案来源 |
|---|---|---|
| efficiency 牌效率 | 10 | 本地 ExactSolver 逐弃牌计算（向听→总受入→种数），含并列备选 |
| defense 防守 | 8 | 确定性安全事实：现物 > 筋/客风客三元 > 无筋中张 |
| push_fold 攻守 | 4 | 标注方向（进攻/防守）+ 求解器复核"保持听牌"约束 |
| value 打点 | 2 | 标注最优弃牌 + 理由关键词 |

难度：basic 11 / advanced 13。所有场景通过物理校验（≤4 张）与"摸牌非自摸"校验。

## 评测方式

- 场景以与生产结构化模式**同一套规则约束**的系统提示词（无图变体）发送，
  low 思考开启，输出契约与生产一致（stance/discard/reason/...）。
- 故意**不提供** local_calculations：测模型自己的牌效率判断，而非照抄程序答案。

## 评分

- efficiency：弃牌 = 期望或备选（并列最优均算对）
- defense：弃牌 ∈ 安全集 且 ∉ 危险集
- push_fold：stance 方向正确，且 fold 时弃牌 ∈ 安全集 / push 时弃牌 ∈ 保持听牌集合
- value：弃牌正确 + 理由含关键事实词（如"宝牌"）

## 用法

```bash
python _dev/benchmark/run_benchmark.py --validate          # 场景自检
python _dev/benchmark/run_benchmark.py --run               # 全量评测（约 24×6s）
python _dev/benchmark/run_benchmark.py --run --ids eff-001,def-001
python _dev/benchmark/run_benchmark.py --run --limit 3     # 冒烟
```

结果明细写入 `_dev/benchmark/results-<时间戳>.json`，含每场景模型原话，便于复盘。

## 维护

- 修改/新增场景：改 `make_scenarios.py` 里的 SCENARIOS 定义 → 重新运行生成
  `scenarios.json`（效率类期望由求解器自动重算）→ `--validate` 通过后提交。
- 场景答案一经提交即为基准基线；改动答案需在提交信息中说明理由。
- 已知边界：牌效率期望不含打点/宝牌权重（效率类场景已规避此冲突）；
  鸣牌判断（call）暂未纳入——结构化输出契约暂无 call 字段。
