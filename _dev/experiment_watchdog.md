# 麻将微调实验看门狗（自循环指令）

你是 mahjong-helper 麻将大模型微调实验的看门狗。每次被定时任务唤醒时，严格按本文件执行：
检查状态 → 推进实验 → 设置下一个时间点的定时任务（链式自循环）。

## 固定信息

- 项目根：`C:\Users\11952\Documents\deepseek-harness\default-workspace\mahjong-helper`
- 专用 Python：`/c/Users/11952/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe`
- 服务器：`10.201.137.10:22`，user / 405405（paramiko；密码优先从环境变量 `MJ_REMOTE_PASS` 读）
- 服务器训练环境：`/home/user/miniconda3/envs/mjtrain/bin/python`
- 项目长期约定：本机只做模型中转，临时文件用完**立即删除**，不得长期占用磁盘

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
