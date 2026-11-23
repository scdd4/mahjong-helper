"""真人策略微调：只学第四层决策（给定结构化事实选 A-N 单字母）。

与前三次失败微调的根本区别：
  - 输入：生产 jev_decision.build_prompt 的完整结构化事实（非原始牌面）
  - 标签：凤凰桌真人实际弃牌映射的 A-N 字母（非手写规则）
  - 输出：单字母（候选封闭，幻觉率 0）
  - 与推理完全同分布：SYSTEM / prompt / 候选构造 / 安全事实全部复用生产代码

架构沿用 train_mahjong.py 的 FastModel + 动态 LoRA target + assistant_only_loss。

用法（服务器）：
    python train_human_policy.py --model /home/user/models/qwen38-4b \
        --data /home/user/mahjong_train/data/human-train.jsonl \
        --out /home/user/mahjong_train/out-human-4b --steps 1500 --rank 16
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')
os.environ.setdefault('HF_HOME', '/home/user/.cache/huggingface')

from jev_decision import SYSTEM

# 生产 assistant 回复就是单个字母（关思考），不加空思考块——保持一致
# 注：Qwen3.8 chat template 可能自动插 <think></think>，推理侧同样会插，故对齐


def discover_lora_targets(model):
    import torch
    skip = ('lm_head', 'visual', 'vision', 'merger', 'embed', 'ngram')
    deltanet_hints = ('in_proj', 'out_proj', 'ba_proj', 'conv', 'query', 'key', 'value')
    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(s in name.lower() for s in skip):
            continue
        targets.add(name.split('.')[-1])
    allowed = {t for t in targets if any(h in t.lower() for h in deltanet_hints)
               or t in ('gate_proj', 'up_proj', 'down_proj', 'o_proj')}
    print('LoRA targets:', sorted(allowed), flush=True)
    if not allowed:
        raise RuntimeError('未发现 LoRA target')
    return sorted(allowed)


def load_data(path, tokenizer, max_len, limit=0):
    from datasets import Dataset
    convs = []
    skipped = 0
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            try:
                row = json.loads(line)
                prompt, letter = row['prompt'], row['letter']
                if not prompt or letter not in 'ABCDEFGHIJKLMN':
                    skipped += 1
                    continue
                convs.append({'messages': [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': letter},
                ]})
            except (ValueError, KeyError, TypeError):
                skipped += 1
            if limit and len(convs) >= limit:
                break
    print('加载 %d 条样本，跳过 %d 条' % (len(convs), skipped), flush=True)
    if not convs:
        raise RuntimeError('数据集为空')
    return Dataset.from_list(convs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seq-len', type=int, default=512)
    ap.add_argument('--save-steps', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig

    # meta tensor 兼容：27B 训练遗留的 device_map 懒加载会让 fix_untrained_tokens
    # 在 model.to() 时崩（Cannot copy out of meta tensor）。禁用该函数。
    try:
        import unsloth_zoo.tokenizer_utils as _tu
        _tu.fix_untrained_tokens = lambda *a, **k: None
    except Exception as _e:
        print('patch tokenizer_utils 跳过: %s' % _e, flush=True)
    # 清理历史编译缓存（27B 训练留下的 UnslothSFTTrainer 缓存会带入旧的 meta 行为）
    import shutil
    cache = Path('/home/user/mahjong_train/unsloth_compiled_cache')
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
        print('已清理 unsloth_compiled_cache', flush=True)

    t0 = time.perf_counter()
    print('加载模型 %s ...' % args.model, flush=True)
    model, processor = FastModel.from_pretrained(
        model_name=args.model, max_seq_length=args.seq_len, dtype=None,
        load_in_4bit=False, full_finetuning=False, ignore_mismatched_sizes=True)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    print('模型加载 %.1fs' % (time.perf_counter() - t0), flush=True)

    targets = discover_lora_targets(model)
    model = FastModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias='none',
        target_modules=targets,
        use_gradient_checkpointing=(os.environ.get('GRAD_CHECKPOINT', 'standard') != 'offload'),
        random_state=3407)

    dataset = load_data(args.data, tokenizer, args.seq_len, args.limit)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=SFTConfig(
            max_seq_length=args.seq_len,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            warmup_steps=50, max_steps=args.steps, learning_rate=args.lr,
            logging_steps=10, save_steps=args.save_steps, save_total_limit=3,
            optim='adamw_8bit', output_dir=args.out, seed=3407, bf16=True,
            report_to='none', packing=False, assistant_only_loss=True))
    print('开始训练：%d 步 / %d 样本' % (args.steps, len(dataset)), flush=True)
    trainer.train()
    print('训练完成 %.1f 分钟' % ((time.perf_counter() - t0) / 60), flush=True)

    model.save_pretrained(args.out + '/final')
    tokenizer.save_pretrained(args.out + '/final')
    print('导出 merged...', flush=True)
    model.save_pretrained_merged(args.out + '/merged', tokenizer, save_method='merged_16bit')
    print('全部完成', flush=True)


if __name__ == '__main__':
    main()
