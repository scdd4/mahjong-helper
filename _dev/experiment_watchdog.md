# 麻将微调实验看门狗（自循环指令）

你是 mahjong-helper 麻将大模型微调实验的看门狗。每次被定时任务唤醒时，严格按本文件执行：
检查状态 → 推进实验 → 设置下一个时间点的定时任务（链式自循环）。

## 固定信息

- 项目根：`C:\Users\11952\Documents\deepseek-harness\default-workspace\mahjong-helper`
- 专用 Python：`/c/Users/11952/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe`
- 服务器：`10.201.137.10:22`，user / 405405（paramiko；密码优先从环境变量 `MJ_REMOTE_PASS` 读）
- 服务器训练环境：`/home/user/miniconda3/envs/mjtrain/bin/python`
- 项目长期约定：本机只做模型中转，临时文件用完**立即删除**，不得长期占用磁盘

## 实验进展记录（每跳更新，最新在上）

### 2026-11-22 23:00 — 全部完成，决策 C 已落实，链路终止
- 27B 评测已完成（22:50 结果落盘，59 场景）。SFTP 拉回 → `rescore_eval.py` 重新提取 →
  独立严格解析复核，两者一致：**27B = 29/59（49%），中位 5.03s；4B = 42/59（71%），中位 0.95s**。
- 27B 反而比 4B 差 22 点、慢 5.3 倍；两代微调均未超越 Jev 基线 81%（48/59，1.24s）。
  幻觉率三方均 0%（出牌均合法）。
- 失败特征：value 0/7、push_fold 1/10，效率题从 4B 的 100% 退化到 62%，
  回复反复出现模板文案（"保留宝牌役牌打浮牌"），属背文案非学决策。
- 已生成 `docs/微调实验报告.md`（结论：维持 Jev 方案，微调不投产），并提交 git。
- 决策树 C 完成：**不再设置下一个定时任务，链式看门狗终止**。
- 服务器无遗留进程，本机暂存 `/tmp/mj-27b` 已清空（4.0K）。

### 2026-11-22 22:38 — 27B 训练完成，评测已启动
- 27B 全量训练 **1200/1200 完成**（20:55），loss 收敛 0.014，checkpoint-400/800/1200 + final 齐全。
- merged 导出完成：`/home/user/mahjong_train/out-27b/merged`（52G，18 分片 + 配置文件），"全部完成"标志命中。
- 训练实际 27.0~27.5s/步，总耗时 ~9.2h（11:35→20:55），与 12:29 修正预估一致。
- 内存 118Gi 可用 / 磁盘 2.6T 空闲。
- **22:38 已启动 27B 评测**（eval_server.py，PID 2004551，日志 `/home/user/mahjong_train/eval-27b.log`，
  结果 `/home/user/mahjong_train/eval-27b.json`）。预计 30-60 分钟。
- 评测完成后：SFTP 拉回 → **必须用 `rescore_eval.py`**（勿信 discard 字段）→ 对比 81% 基线 → 决策 C 出报告。
- 注：13:29 一跳未见执行记录（watchdog memory 停更于 12:29），训练无人值守跑完无异常，视为该跳遗漏而非故障。
- 下一跳：2026-11-22T23:30（评测进行中）。

### 2026-11-22 12:29 — 27B 训练巡检
- 27B 全量训练运行中：**101/1200（8.4%）**，loss 2.686→0.047 正常下降，无报错。
- **速度修正（重要）**：实际 ~27.5s/步（非 smoke 的 9.5s/步），全量 1200 步预计 **~9.2 小时**（非 3.2h），约 20:45 完成训练 + ~7min 合并导出。日志尾部"预计 30 分钟"为脚本占位字符串，勿采信。
- meta tensor patch 生效（日志命中 1 次）；训练进程 PID 2001533 存活（11:35 启动，rank 16）。
- 内存 61Gi 可用 / 磁盘 2.7T 空闲，均充足（27B 占用约 60Gi 内存，符合预期）。
- 本地 dl27b 完成、暂存清零；auto_eval_4b 日志为已知超时 traceback（结果已产出，无需处理）。
- 决策 A/B 已完成、C 未触发；未停止/删除任何进程。
- 下一跳：2026-11-22T13:29（训练进行中且进度<90% → +60min），链式自动化 id=bb737659… 已更新。

### 2026-11-22 10:19 — 看门狗巡检
- 4B：完成标志=1，eval-4b.json 已评分（42/59 / 71%，未超越 81% 基线），决策 A 已完成，本地副本在 `_dev/benchmark/eval-4b-server.json`。
- 27B：15/18 分片（43G），第 16 片传输中 71.7%；本机暂存 2.8G 为在途分片，下载进程活跃、日志持续刷新（符合预期，非滞留）。
- 内存 117Gi 可用 / 磁盘 2.7T 空闲，均充足。
- 决策 B/C 未触发；未停止或删除任何进程。`auto_eval_4b.py` 已退出（超时 traceback），但评测结果已产出并评分，无需处理。
- 下一跳：2026-11-22T11:19（下载 <90% → +60min），链式自动化 id=bb737659… 已更新。

### 2026-11-22 11:35 — 27B 全量训练已启动

27B 下载完成（18/18 分片，52G，本机暂存清零）。smoke 验证 2 步成功
（loss 3.042，9.52s/步，Trainable 109M/27.5B=0.40%），已启动全量 1200 步。

**27B 训练的已知坑（已修）**：`fix_untrained_tokens` 遇 device_map 懒加载的
meta tensor 报 `Cannot copy out of meta tensor`。已在训练脚本开头 monkeypatch 掉，
启动前删除 `unsloth_compiled_cache` 让其重新编译。日志出现
"已禁用 fix_untrained_tokens（27B meta tensor 兼容）"表示 patch 生效。

**27B 全量训练参数**：rank 16 / batch 1 / accum 16 / seq 512 / lr 2e-4 / 1200 步
日志 `/home/user/mahjong_train/train-27b.log`，产物 `/home/user/mahjong_train/out-27b/merged`
预计约 3.2 小时（9.5s/步）+ 合并导出约 7 分钟。

**27B 评测完成后**：`eval_server.py`（--model 指向 out-27b/merged）→
**必须用 `rescore_eval.py` 重新提取牌名**（eval_server 的 discard 字段解析有 bug，
reply 字段完整）→ 对比基线 81%。

### 2026-11-22 09:40 — 4B 评测完成

**4B 微调结果：42/59 (71%)，中位延迟 0.95s。比当前 Jev 基线 81% 低 10 个百分点。**

分类：efficiency 21/21 (100%) / defense 17/21 (81%) / push_fold 3/10 (30%) / value 1/7 (14%)

关键结论：
- 格式修复成功（assistant_only_loss 生效），模型能正确输出 `牌名|理由`
- 牌效率完美（100%），说明求解器候选 + 微调对纯效率判断非常有效
- **短板是 value(14%) 和 push_fold(30%)**：4B 参数量不足以学会宝牌打点取舍与攻守权衡
- 训练数据的 value 类理由是模板化文案（"保留宝牌役牌打浮牌"），没有教具体打点计算

4B 未超越基线。27B 是下一个希望（参数量 6.75 倍，可能补上打点/攻守推理）。

**注意**：`rescore_eval.py` 已修复解析（用正则 `([1-9][mpsz])\s*\|` 提取，因 skip_special_tokens 会剥掉 think 标签尖括号）。
评测脚本 `eval_server.py` 的 discard 提取仍有旧 bug，但 reply 字段完整，用 rescore 即可。
**27B 评测必须用 rescore_eval.py 重新提取，不要直接信 eval_server 输出的 discard 字段。**

### 2026-11-22 09:30 — 4B 训练完成
1200 步完成，loss 0.020，merged 8.7G 已导出到 `/home/user/mahjong_train/out-4b/merged`。

## 实验背景（2026-11-22 启动）

两步走全量微调，把 Spark 上的 Qwen 微调为雀魂麻将决策模型。数据 41503 条（训练 37352 + 验证 4151），
答案由 ExactSolver + 安全规则生成。评测口径：59 场景基准，当前基线（Jev 关思考）81%。

关键修复：第一版训练未开 `assistant_only_loss`，模型只学会复述 prompt，已改为对话式数据集 +
assistant_only_loss=True + seq512 + 答案前显式空思考块，10 步冒烟验证 loss 正常下降。

## 后台任务（已在运行）

1. **4B 全量训练**：1200 步，日志 `/home/user/mahjong_train/train-4b.log`，完成标志字符串 `全部完成`，
   产物 `/home/user/mahjong_train/out-4b/merged`
2. **4B 自动评测**：本机后台 `auto_eval_4b.py`（日志 `/tmp/auto_eval_4b.log`）会在训练完成后自动跑
   `eval_server.py`，结果写 `/home/user/mahjong_train/eval-4b.json`，本机日志 `/tmp/auto_eval_4b.log`
3. **27B 下载中转**：本机后台 `download_27b.py`（日志 `/tmp/dl27b.log`），18 个分片到
   `/home/user/models/qwen38-27b/`，每片传完自动清理本机暂存（`$TEMP/mj-27b`）

## 每次唤醒的检查清单

用 paramiko 连服务器，依次检查（命令用 `exec_command`）：

```
grep -oE '[0-9]+/1200' /home/user/mahjong_train/train-4b.log | tail -1     # 4B 训练进度
grep -c '全部完成' /home/user/mahjong_train/train-4b.log                    # 4B 是否完成
ls -lh /home/user/mahjong_train/eval-4b.json 2>/dev/null                   # 4B 评测是否出结果
ls -1 /home/user/models/qwen38-27b/*.safetensors 2>/dev/null | wc -l       # 27B 分片数(满18=完成)
free -h | head -2                                                          # 内存(27B训练需>80G可用)
```

本机检查：

```
tail -5 /tmp/dl27b.log                    # 27B 下载日志
tail -8 /tmp/auto_eval_4b.log             # 4B 自动评测日志
du -sh "$TEMP/mj-27b" 2>/dev/null         # 本机暂存(必须持续下降，传完即删)
```

## 推进决策树

**A. 4B 评测结果已生成**（`eval-4b.json` 存在）：
1. SFTP 下载到项目 `_dev/benchmark/eval-4b-server.json`
2. 运行评分：`python _dev/score_eval.py _dev/benchmark/eval-4b-server.json --name 4B微调 --baseline 0.81`
3. 记录准确率、分类准确率、延迟。判断是否 >81%（当前 Jev 基线）：
   - 若明显低于基线 → 在报告里如实记录"微调未超越基线"，分析原因（可能 assistant_only_loss 后仍需更多步、数据质量、基座能力）
   - 若显著高于基线 → 记录为有效方案

**B. 27B 下载完成**（18 片齐全）：
启动 27B 全量训练（8080 服务已停，内存充足）。命令模板：

```
cd /home/user/mahjong_train && setsid /home/user/miniconda3/envs/mjtrain/bin/python -u train_mahjong.py \
  --model /home/user/models/qwen38-27b \
  --data /home/user/mahjong_train/data/train.jsonl \
  --out /home/user/mahjong_train/out-27b \
  --steps 1200 --rank 32 --batch 1 --accum 16 --lr 2e-4 --seq-len 512 --save-steps 400 \
  > /home/user/mahjong_train/train-27b.log 2>&1 < /dev/null &
```

注意：27B 用 batch 1（内存保守），若 OOM 降到 rank 16。先跑 2 步 smoke 验证再全量。
训练完成后同样用 `eval_server.py` 评测（--model 指向 out-27b/merged），`score_eval.py` 评分。

**C. 全部完成**（4B 评测 + 27B 评测都有结果）：
生成完整简洁实验报告 `docs/微调实验报告.md`，内容：
- 结论先行：微调是否超越当前 Jev 方案（81% / 1.2s）
- 对比表：当前 Jev / 4B 微调 / 27B 微调 的准确率、分类准确率、延迟、幻觉率
- 数据方案有效性、训练中修复的问题、交付物清单（模型路径、脚本、数据）
- 提交 git：`git add && git commit && git push`
- **不再设置下一个定时任务**（链路终止）

## 设置下一个定时任务（关键，每次唤醒必做，除非全部完成）

调用 `automation_update`：
- `mode`: "create"
- `scheduleType`: "once"
- `name`: "麻将微调实验看门狗（链式）"
- `prompt`: 固定为下面这一句（原样复制，保证下一跳能续链）：
  `读取 C:\Users\11952\Documents\deepseek-harness\default-workspace\mahjong-helper\_dev\experiment_watchdog.md 并严格遵循其中指令推进实验。`
- `cwds`: `C:\Users\11952\Documents\deepseek-harness\default-workspace\mahjong-helper`
- `scheduledAt`: 下一个时间点（ISO 8601 本地时间 GMT+8），按规则选：
  - 训练/下载进行中且进度 < 90% → 当前时间 + 60 分钟
  - 进度 ≥ 90% 或评测运行中 → + 15 分钟
  - 4B 评测刚出结果、需启动 27B 训练 → + 15 分钟
  - 只有在"全部完成"（决策树 C）时才不设置下一个

用系统时间工具或 `date` 命令获取当前时间再推算，不要凭记忆猜时间戳。

## 约束

- 密码不写入代码/不提交 git，只从环境变量或用户提供的凭据取
- 本机模型临时文件用完立即删除，每次检查暂存目录确认在收缩
- 报告简洁：结论先行，避免过程流水账
- 不擅自停止/删除正在运行的训练或下载进程
