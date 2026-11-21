"""服务器离线评测：加载 merged 微调模型，用训练 prompt 格式跑 59 场景基准。

在服务器上运行（GPU 空闲时）：
    python eval_server.py --model /home/user/mahjong_train/out-4b/merged \
        --scenarios /home/user/mahjong_train/data/scenarios.json \
        --out /home/user/mahjong_train/eval-4b.json

输出每题 discard + 理由，本机再用 score() 评分（评分逻辑不动，保证口径与历史可比）。
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_SFT = ('你是雀魂麻将决策模型。只输出一个牌名和一句≤12字理由，格式：牌名|理由。'
              '依据局面事实（安全集、立直、役牌、宝牌、点棒、巡目、向听与进张）选择最优弃牌。')

SEATS = ('own', 'right', 'across', 'left')


def safety_from(state):
    """复用 gen_training_data 的安全事实口径：现物/筋 + 共同安全。"""
    riichi = state.get('riichi') or {}
    rivers = state.get('rivers') or {}
    active = [s for s, on in riichi.items() if on and s != 'own']
    genbutsu, suji = {}, {}
    for seat in active:
        river = rivers.get(seat) or []
        genbutsu[seat] = list(dict.fromkeys(river))
        suji[seat] = []
        for t in river:
            if t.endswith('m') or t.endswith('p') or t.endswith('s'):
                n = int(t[0])
                if 1 <= n <= 6:
                    suji[seat].append('%dm' % (n + 3) if t.endswith('m') else
                                      '%dp' % (n + 3) if t.endswith('p') else '%ds' % (n + 3))
        suji[seat] = list(dict.fromkeys(suji[seat]))
    common = set(state['hand'])
    for seat in active:
        common &= (set(genbutsu[seat]) | set(suji[seat]))
    return {'现物': genbutsu, '筋': suji}, sorted(common)


def format_prompt(state):
    """与 gen_training_data.format_prompt 完全一致的训练格式。"""
    hand = state['hand']
    lines = ['手牌(%d张)：%s' % (len(hand), ' '.join(hand))]
    calc = state.get('local_calculations') or {}
    if calc.get('shanten') is not None:
        lines.append('当前向听：%s' % calc['shanten'])
    for i, c in enumerate(calc.get('candidates') or []):
        letter = chr(ord('A') + i)
        note = []
        if c.get('shanten') is not None:
            note.append('%s向听' % c['shanten'])
        if c.get('ukeire'):
            note.append('%s张进张' % c['ukeire'])
        if c.get('tenpai'):
            note.append('听牌')
        lines.append('%s=%s（%s）' % (letter, c['tile'], '，'.join(note)) if note else '%s=%s' % (letter, c['tile']))
    riichi = state.get('riichi') or {}
    for seat, on in riichi.items():
        if on and seat != 'own':
            lines.append('%s立直' % seat)
    rivers = state.get('rivers') or {}
    for seat in SEATS[1:]:
        if rivers.get(seat):
            lines.append('%s牌河：%s' % (seat, ' '.join(rivers[seat][:12])))
    safety, common = safety_from(state)
    if common:
        lines.append('对所有立直家共同安全：' + ' '.join(common))
    for kind in ('现物', '筋'):
        for seat, tiles in (safety.get(kind) or {}).items():
            if tiles:
                lines.append('%s对%s：%s' % (kind, seat, ' '.join(tiles[:8])))
    if state.get('dora_indicators'):
        lines.append('宝牌指示牌：' + ' '.join(state['dora_indicators']))
    if state.get('round'):
        lines.append('场风：%s' % state['round'][:1])
    if state.get('self_wind'):
        lines.append('自风：%s' % state['self_wind'])
    if state.get('scores'):
        lines.append('点棒：' + ' '.join('%s=%s' % (k, v) for k, v in state['scores'].items()))
    if state.get('turn'):
        lines.append('巡目：%s' % state['turn'])
    lines.append('轮到你出牌。只输出牌名|理由。')
    return '\n'.join(lines)


def scenario_state(sc):
    """把基准场景转成带候选/向听的 state（与 eval_finetuned 同口径）。"""
    state = dict(sc['state'])
    calc = state.get('local_calculations') or {}
    if not calc and sc.get('solver_table'):
        cands = sorted(sc['solver_table'], key=lambda x: (x['shanten'], -x['ukeire']))[:8]
        shanten = min((r['shanten'] for r in sc['solver_table']), default=None)
        calc = {'shanten': shanten,
                'candidates': [{'tile': r['drop'], 'shanten': r['shanten'], 'ukeire': r['ukeire']}
                               for r in cands]}
    state['local_calculations'] = calc
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--scenarios', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-new', type=int, default=48)
    args = ap.parse_args()

    print('加载模型 %s ...' % args.model, flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.eval()
    print('模型加载 %.1fs, device=%s' % (time.perf_counter() - t0, model.device), flush=True)

    scenarios = json.loads(Path(args.scenarios).read_text(encoding='utf-8'))['scenarios']
    results = []
    for sc in scenarios:
        state = scenario_state(sc)
        user = format_prompt(state)
        messages = [{'role': 'system', 'content': SYSTEM_SFT}, {'role': 'user', 'content': user}]
        # 训练时移除了空思考块，模型学到 assistant\n 后直接跟答案。
        # 评测必须同样去掉 <think>\n，否则模型进入思考模式复述 prompt。
        text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_in = text_in.replace('<think>\n', '')
        inputs = tok(text_in, return_tensors='pt').to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        gen = out[0][inputs['input_ids'].shape[1]:]
        reply = tok.decode(gen, skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - t0
        # Qwen3.8 会在 assistant 后固化生成 <think>；训练时该块被移除，
        # 答案在 </think> 之后。若模型仍进入思考，剥掉思考块取答案。
        if '</think>' in reply:
            reply = reply.split('</think>')[-1].strip()
        elif reply.startswith('<think>'):
            reply = reply[len('<think>'):].strip()
        discard = reply.split('|')[0].strip().split()[0] if reply else ''
        results.append({'id': sc['id'], 'kind': sc['kind'], 'discard': discard,
                        'reply': reply, 'seconds': round(elapsed, 2)})
        print('%-10s %-8s %s  %.2fs  %s' % (sc['id'], sc['kind'], discard, elapsed, reply[:40]), flush=True)

    Path(args.out).write_text(json.dumps({'model': args.model, 'results': results},
                                         ensure_ascii=False, indent=2), encoding='utf-8')
    print('写入 %s' % args.out, flush=True)


if __name__ == '__main__':
    main()
