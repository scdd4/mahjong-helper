"""验证修复后的 eval_server 提示词口径：候选行齐全、无伪共同安全行。"""
import json
import sys

sys.path.insert(0, '/home/user/mahjong_train')
from eval_server import format_prompt, scenario_state

scenarios = json.load(open('/home/user/mahjong_train/data/scenarios-v2.json', encoding='utf-8'))['scenarios']
n_cand = n_fake_common = 0
for sc in scenarios:
    prompt = format_prompt(scenario_state(sc))
    if 'A=' in prompt:
        n_cand += 1
    has_riichi = any(l in prompt for l in ('right立直', 'across立直', 'left立直'))
    if '对所有立直家共同安全' in prompt and not has_riichi:
        n_fake_common += 1
print('带候选行场景: %d/59' % n_cand)
print('无立直却标共同安全的伪影: %d' % n_fake_common)
d = [s for s in scenarios if s['id'] == 'def-011'][0]
print('--- def-011 修复后提示词 ---')
print(format_prompt(scenario_state(d)))
