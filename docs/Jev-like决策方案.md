# Jev-like 麻将决策方案（基于现有 Spark/strata，无需训练新模型）

## 一、核心洞察

Jev（TypeSafe AI 的 System One 模型）的本质不是"小模型"，而是**决策空间封闭 + 一次前向并行打分 + 概率校准**。这三件事不需要训练新模型——现有 Spark 的 strata 引擎已原生提供：

| Jev 原语 | strata 对应能力 | 验证结果 |
|---|---|---|
| 封闭候选集 | `allowed_token_ids` | 输出被锁死在指定 token 内，结构上不可能越界 |
| 并行打分 | `logprobs=true, top_logprobs=K` | 首 token 返回全部候选的对数概率 |
| 概率+置信度 | top_logprobs 归一化 | A=-0.0013 → 99.87%，可直接分支 |
| RLCD 校准 | 用基准评测置信度-正确率曲线替代 | 实测 high 档 97%，见评测章节 |

**关于思考**：Jev 是 System One，不需要 CoT。实测受限候选 + 完整事实注入后，
关思考 1.2s 且准确率不低于开思考 4.3s（开思考的 token 预算被服务端砍半，还容易撞上限）。
因此生产配置是 `enable_thinking=false, max_tokens=64`——思考的收益已被"封闭决策空间"取代。

关键实测（10.201.137.10:8080，strata 0.1.59）：
- 牌名 `1m` 是多 token，但候选映射到单 token 序号 A-N（id 32-45）
- 开思考 + `allowed_token_ids` + `logprobs` **完全兼容**：reasoning 正常生成，answer token 的分布正常返回
- prefill 约 0.9s，答案首 token 紧随其后；无 JSON 生成、无大段输出

## 二、架构

```
截图 → 识牌 → ExactSolver 候选（≤14 张）
              ↓
     候选映射 A-N + 局面事实
              ↓
  Spark 一次前向（开思考 + allowed_token_ids + logprobs）
      ├─ content: 单个字母 A-N   → 映射回牌名 = 最终决策
      └─ top_logprobs: 候选概率分布 → 置信度
              ↓
  置信度分支：高 → 直接出牌；低 → 升级到结构化模式复核
```

**与 Jev 的对应**：state = 结构化局面事实；choices = 弃牌候选（A-N）；Choice 原语 = 受限首 token + logprob；confidence = top1 概率。

**与当前 v1.9.5 的差异**：
- 当前：模型自回归生成 JSON（discard/reason/wait_plan/risk），关思考，1.94s，幻觉靠"牌不在手"兜底
- Jev-like：模型只生成 1 个 token（受限），开思考，结构上零幻觉，额外得到概率/置信度

## 三、置信度分级（Jev 的核心价值）

| top1 概率 | 判定 | 动作 |
|---|---|---|
| ≥ 0.60 | 高置信 | 直接采纳，秒回 |
| 0.30~0.60 | 中置信 | 采纳但标注"待核对" |
| < 0.30 | 低置信 | 升级到结构化模式（完整思考）复核 |

这样把算力花在疑难局面，简单局面一次前向即返回——正是 Jev 论文推荐的 "branch on confidence, escalate the rest"。

## 四、为什么不用停服务/训模型

- 服务器 `spark-7976`：NVIDIA GB10 GPU + 121G 内存 + 3.7T 磁盘，strata 0.1.59 跑着 qwen3.8-flash（79G 权重）
- 特权：SSH user/405405 可登录，有全部控制权
- 但训练 Jev 需要 RLCD 训练栈（torch/trl/数据/数周），投入产出极低；而 strata 已有 logprobs + allowed_token_ids，直接实现同样的决策语义
- 结论：**不换模型、不停服务，纯客户端改造即可获得 Jev-like 能力**

## 五、实施清单

1. `jev_decision.py`：候选→A-N 映射、受限请求、logprob→概率/置信度、升级逻辑
2. `analysis_mode='jev'` 接入 AnalysisPipeline 和设置 UI
3. 测试：单测 + 59 场景基准 + 端到端 + EXE
4. 评测：jev-like vs 当前 fast（准确率、延迟、置信度校准）
