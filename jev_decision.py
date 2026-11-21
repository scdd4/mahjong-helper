"""Jev-like 麻将决策：封闭候选集 + 一次前向 + logprobs 概率分布。

原理（见 docs/Jev-like决策方案.md）：
- 把最多 14 个弃牌候选映射到单 token 序号 A-N（vocab id 32-45）。
- 用 strata 的 allowed_token_ids 把首 token 锁死在候选内——结构上不可能输出
  不存在的牌，无需事后幻觉兜底。
- logprobs=true + top_logprobs 返回候选的概率分布，top1 即置信度，可分级分支。
- 思考在服务端照常进行（reasoning_content 不展示给用户），用户只看到一张牌。
"""
import math
import time

# A-N 在 qwen3.8 tokenizer 中是连续 id（已实测）
TOKEN_BASE = ord('A')
TOKEN_FIRST_ID = 32
MAX_CHOICES = 14

SYSTEM = '''你是四人立直麻将决策模型。用户给出结构化局面事实与编号候选，你只输出一个字母代表选择。
【铁律】只输出 A-N 中的一个大写字母，不输出牌名、理由、JSON、标点或任何其它字符。
依据局面事实（安全集、立直家、役牌、宝牌、点棒、巡目、向听与进张）独立选择最优弃牌：
有对手立直且我方无听牌威胁时优先安全牌；末巡落后需点数时保持进攻；牌效率作为基础参考而非唯一标准。
候选含义见输入，字母与牌的对应绝不改变。'''


def candidates_from_calculations(calculations, verified, evidence=None):
    """从本地核算结果构造候选弃牌列表（最多 14 个，去重，必须在手牌里）。

    优先级：ExactSolver 的 candidates（按进张排序）→ 手牌里其余牌补位。
    """
    if not isinstance(calculations, dict):
        return []
    seen = []
    for item in (calculations.get('candidates') or []):
        if isinstance(item, dict):
            tile = item.get('tile')
            if tile and tile in verified and tile not in seen:
                seen.append(tile)
    for tile in verified:
        if tile not in seen:
            seen.append(tile)
    # 安全牌优先候选中已含；这里只截断，不打乱 Solver 的排序
    return seen[:MAX_CHOICES]


def build_prompt(verified, calculations, evidence, candidates):
    """构造给用户角色的紧凑局面事实文本。"""
    lines = ['手牌(%d张)：%s' % (len(verified), ' '.join(verified))]
    if calculations:
        shanten = calculations.get('shanten')
        if shanten is not None:
            lines.append('当前向听：%s' % shanten)
        by_tile = {c.get('tile'): c for c in (calculations.get('candidates') or []) if isinstance(c, dict)}
        for index, tile in enumerate(candidates):
            c = by_tile.get(tile, {})
            note = []
            if c.get('shanten') is not None:
                note.append('%s向听' % c['shanten'])
            if c.get('ukeire') is not None:
                note.append('%s张进张' % c['ukeire'])
            if note:
                lines.append('%s=%s（%s）' % (chr(TOKEN_BASE + index), tile, '，'.join(note)))
            else:
                lines.append('%s=%s' % (chr(TOKEN_BASE + index), tile))
    else:
        for index, tile in enumerate(candidates):
            lines.append('%s=%s' % (chr(TOKEN_BASE + index), tile))
    riichi = []
    for seat, on in ((evidence or {}).get('riichi') or {}).items():
        if on and seat != 'own':
            riichi.append(seat)
    if riichi:
        lines.append('立直家：' + '/'.join(riichi))
    rivers = (evidence or {}).get('rivers') or {}
    for seat in riichi:
        tiles = [t.get('name') for t in (rivers.get(seat) or []) if isinstance(t, dict) and t.get('name')]
        if tiles:
            lines.append('%s牌河：%s' % (seat, ' '.join(tiles)))
    indicators = [v for v in ((evidence or {}).get('dora_indicators') or []) if isinstance(v, str)]
    if indicators:
        lines.append('宝牌指示牌：' + ' '.join(indicators))
    melds = ((evidence or {}).get('melds') or {}).get('own') or {}
    groups = [g for g in (melds.get('groups') or []) if g]
    if groups:
        lines.append('我方副露：' + str(len(groups)) + '组')
    # 安全事实：现物/筋/共同安全牌——攻守判断的关键，缺了模型只能猜。
    safety = (evidence or {}).get('safety_facts') or {}
    for kind in ('现物', '筋', '两筋'):
        by_seat = safety.get(kind) or {}
        for seat, tiles in by_seat.items():
            if tiles:
                lines.append('%s对%s：%s' % (kind, seat, ' '.join(tiles[:8])))
    common = (evidence or {}).get('common_safe_for_all_riichi') or []
    if common:
        lines.append('对所有立直家共同安全：' + ' '.join(common))
    unsafe = (evidence or {}).get('unsafe_hand_tiles') or []
    if unsafe:
        lines.append('危险候选（勿打）：' + ' '.join(unsafe[:10]))
    for key in ('round', 'self_wind', 'remaining', 'dora_tiles', 'yakuhai_tiles'):
        val = (evidence or {}).get(key)
        if val:
            if isinstance(val, (str, int, float)):
                text_val = str(val)
            elif isinstance(val, (list, tuple, set)):
                text_val = ' '.join(map(str, val))
            elif isinstance(val, dict):
                text_val = ' '.join('%s=%s' % (k, v) for k, v in val.items())
            else:
                text_val = str(val)
            lines.append('%s：%s' % (key, text_val))
    lines.append('轮到你出牌。只输出一个字母。')
    return '\n'.join(lines)


def parse_logprobs(choice, candidates):
    """从 answer token 的 top_logprobs 提取候选牌的概率与置信度。

    strata 的 logprobs 格式：
      content[0].top_logprobs = [{token:'A', logprob:-0.001}, ...]
    allowed_token_ids 保证只有候选 token 出现；未列出的候选概率极低，
    用 log(1 - sum(列出概率)) 的均分作保守估计。
    """
    entry = ((choice.get('logprobs') or {}).get('content') or [{}])[0]
    entries = entry.get('top_logprobs') or []
    letter_lp = {}
    for item in entries:
        token = (item.get('token') or '').strip()
        if len(token) == 1 and 'A' <= token.upper() <= 'N':
            lp = item.get('logprob')
            if isinstance(lp, (int, float)):
                letter_lp[token.upper()] = float(lp)
    probs = {}
    for index, tile in enumerate(candidates):
        letter = chr(TOKEN_BASE + index)
        probs[tile] = math.exp(letter_lp[letter]) if letter in letter_lp else 1e-9
    total = sum(probs.values())
    if total > 0:
        probs = {tile: p / total for tile, p in probs.items()}
    chosen = (entry.get('token') or '').strip().upper()
    if len(chosen) != 1 or not ('A' <= chosen <= 'N'):
        chosen = ''
    index = ord(chosen) - TOKEN_BASE if chosen else -1
    tile = candidates[index] if 0 <= index < len(candidates) else None
    confidence = max(probs.values()) if probs else 0.0
    return tile, confidence, probs


def confidence_level(confidence):
    """Jev 式置信度分级：高置信直接采纳，低置信升级复核。"""
    if confidence >= 0.60:
        return 'high'
    if confidence >= 0.30:
        return 'medium'
    return 'low'


def build_report(tile, confidence, probs, candidates, verified, evidence,
                 hand_source, notes, elapsed, model, reasoning_used=False):
    """组装与现有流程兼容的 report。正文极简：只要打什么牌。"""
    observed = {
        'hand': list(verified), 'round': None, 'self_wind': None, 'remaining': None,
        'scores': {}, 'dora_indicators': [v for v in ((evidence or {}).get('dora_indicators') or [])
                                          if isinstance(v, str)], 'dora_tiles': [],
        'riichi': {}, 'rivers': {}, 'melds': {},
    }
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    alternatives = [{'tile': other, 'when': '次选', 'reason': '%.0f%%概率' % (p * 100)}
                    for other, p in ranked if other != tile]
    strategy = {
        'stance': 'Jev决策',
        'recommendation': {'action': '打牌' if tile else '等待摸牌', 'tile': tile,
                           'reason': '模型概率 %.0f%%' % (confidence * 100)},
        'offense': '', 'defense': '',
        'alternatives': alternatives,
        'next_turn_plan': [], 'call_plan': '', 'placement': '',
        'next_discard': '',
        'uncertainties': list(notes),
        'confidence': confidence_level(confidence),
    }
    return {
        'report': {'observed': observed, 'strategy': strategy},
        'hand_source': hand_source,
        'vision_uncertainties': list(notes),
        'thinking_stages': {'jev': 'on'},
        'elapsed_seconds': elapsed,
        'stage_seconds': {'jev': elapsed},
        'stage_usage': {'jev': {}},
        'mode': 'jev',
        'jev_confidence': round(confidence, 4),
        'jev_probabilities': {k: round(v, 4) for k, v in probs.items()},
        'jev_candidates': list(candidates),
        'jev_model': model,
        'jev_reasoning_used': reasoning_used,
    }
