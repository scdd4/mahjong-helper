"""Qwen3.8 雀魂麻将微调训练脚本（Unsloth + LoRA）。

要点：
- Qwen3.8 是 qwen3_5 架构 + 混合 DeltaNet，用 FastModel（非 FastLanguageModel）
- LoRA target 必须动态发现（DeltaNet 不用 q/k/v_proj 命名）
- loss masking：只在 assistant 回复上算 loss
- 定期存 checkpoint 便于评测对比

用法（服务器上）：
    python train_mahjong.py --model /home/user/models/qwen38-4b \
        --data _dev/train_data/train.jsonl --out /home/user/mahjong-lora-4b --steps 3000
"""
import argparse
import json
import os
import time
from pathlib import Path

# Qwen3.8 DeltaNet 关键环境变量（必须在 import torch 之前设置）
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')
os.environ.setdefault('HF_HOME', '/home/user/.cache/huggingface')


def discover_lora_targets(model):
    """动态发现 LoRA target：包含标准 Linear + DeltaNet 投影。"""
    import torch
    skip = ('lm_head', 'visual', 'vision', 'merger', 'embed', 'ngram')
    targets = set()
    deltanet_hints = ('in_proj', 'out_proj', 'ba_proj', 'conv', 'query', 'key', 'value')
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(s in name.lower() for s in skip):
            continue
        short = name.split('.')[-1]
        targets.add(short)
    # 只保留 Unsloth/PEFT 支持的类型
    allowed = {t for t in targets if any(h in t.lower() for h in deltanet_hints)
               or t in ('gate_proj', 'up_proj', 'down_proj', 'o_proj')}
    print('发现 LoRA targets:', sorted(allowed), flush=True)
    if not allowed:
        raise RuntimeError('未发现任何 LoRA target，架构可能不兼容')
    return sorted(allowed)


def load_data(path, tokenizer, max_len):
    """读取 JSONL 为对话式数据集，配合 assistant_only_loss=True。

    第一版用整段 text 训练（未开 assistant_only_loss），模型只学会复述 prompt，
    没学到"assistant 后直接出牌名|理由"——评测证实输出全是局面分析。
    改为标准 messages 列表，trl 的 assistant_only_loss 自动只对 assistant 回合算 loss。

    Qwen3.8 会在 assistant 回合固化插入空思考块 <think></think>，
    训练答案里显式写空思考块以对齐基座分布：assistant 先输出空思考，再出牌名|理由。
    """
    from datasets import Dataset
    convs = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            try:
                messages = json.loads(line)['messages']
                if len(messages) != 3:
                    continue
                sys_c = messages[0]['content']
                usr_c = messages[1]['content']
                answer = messages[-1]['content']
                if not (sys_c and usr_c and answer):
                    continue
                # 对齐基座：assistant 回合带空思考块，之后才是答案
                assistant = '<think>\n\n</think>\n\n' + answer
                convs.append({'messages': [
                    {'role': 'system', 'content': sys_c},
                    {'role': 'user', 'content': usr_c},
                    {'role': 'assistant', 'content': assistant},
                ]})
            except (ValueError, KeyError, IndexError, TypeError):
                continue
    print('加载 %d 条样本（对话式, assistant_only_loss）' % len(convs), flush=True)
    if not convs:
        raise RuntimeError('数据集为空，请检查数据文件')
    return Dataset.from_list(convs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--limit', type=int, default=0, help='只用前N条样本（调试用）')
    args = parser.parse_args()

    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig

    t0 = time.perf_counter()
    print('加载模型 %s ...' % args.model, flush=True)
    model, processor = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seq_len,
        dtype=None,
        load_in_4bit=False,
        full_finetuning=False,
        # qwen3_5 多模态权重里有视觉塔参数，纯文本微调时忽略这些不匹配的键
        ignore_mismatched_sizes=True,
    )
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    print('模型加载完成 %.1fs' % (time.perf_counter() - t0), flush=True)

    targets = discover_lora_targets(model)
    model = FastModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        bias='none',
        target_modules=targets,
        # 内存充足时用标准 checkpoint（True）避免 Unsloth 的梯度 offload（会拖慢约 3 倍）
        use_gradient_checkpointing=(os.environ.get('GRAD_CHECKPOINT', 'standard') != 'offload'),
        random_state=3407,
    )

    dataset = load_data(args.data, tokenizer, args.seq_len)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print('调试：截断到 %d 条' % len(dataset), flush=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            max_seq_length=args.seq_len,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            warmup_steps=50,
            max_steps=args.steps,
            learning_rate=args.lr,
            logging_steps=10,
            save_steps=args.save_steps,
            save_total_limit=3,
            optim='adamw_8bit',
            output_dir=args.out,
            seed=3407,
            bf16=True,
            report_to='none',
            packing=False,
            # 关键：只对 assistant 回复算 loss，否则模型只学会复述 prompt
            assistant_only_loss=True,
        ),
    )
    print('开始训练：%d 步，总样本 %d，预计 %.0f 分钟' %
          (args.steps, len(dataset), args.steps * 1.5 / 60), flush=True)
    trainer.train()
    print('训练完成，总耗时 %.1f 分钟' % ((time.perf_counter() - t0) / 60), flush=True)

    # 保存最终 adapter
    model.save_pretrained(args.out + '/final')
    tokenizer.save_pretrained(args.out + '/final')
    print('已保存 adapter 到 %s/final' % args.out, flush=True)

    # 合并导出 safetensors（strata/vLLM 可直接加载，比 GGUF 转换更可靠）
    print('导出 merged safetensors...', flush=True)
    merged = args.out + '/merged'
    model.save_pretrained_merged(merged, tokenizer, save_method='merged_16bit')
    print('已导出 %s' % merged, flush=True)

    # GGUF 导出可选：需要现场编译 llama.cpp，依赖 apt 包与外网，单独处理
    if os.environ.get('EXPORT_GGUF') == '1':
        print('导出 GGUF q8_0...', flush=True)
        model.save_pretrained_gguf(args.out + '/gguf', tokenizer, quantization_method='q8_0')
        print('GGUF 完成', flush=True)
    print('全部完成', flush=True)


if __name__ == '__main__':
    main()
