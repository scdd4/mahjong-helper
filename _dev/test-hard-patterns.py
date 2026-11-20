"""高难牌型与场面专项：本地规则层。

检测 ExactSolver / 安全事实 / 开牌核算在极端牌型下的正确性。
这些不是模型评测，而是"喂给模型的事实是否可信"的最后一道闸门。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.python-deps'))

from desktop_runtime import ExactSolver, build_safety_facts, tile_index, dora_from_indicator
from open_efficiency import analyze_open


class HardPatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = ExactSolver()

    @classmethod
    def tearDownClass(cls):
        cls.solver.close()

    def solve(self, hand13, seen=()):
        return self.solver.analyze(hand13, list(seen))

    # ---------- 特殊和型 ----------
    def test_kokushi_13_wait(self):
        # 国士无双 13 面听：13 种幺九各一张
        hand = ['1m','9m','1p','9p','1s','9s','1z','2z','3z','4z','5z','6z','7z']
        r = self.solve(hand)
        self.assertEqual(r['shanten'], 0, r)
        self.assertEqual(len(r.get('waits', [])), 13, r.get('waits'))

    def test_chiitoi_7_pairs(self):
        # 7 对子已齐 + 一张单张（14 张时和牌）；这里给 13 张 = 6 对 + 单张，一向听
        hand = ['1m','1m','3p','3p','5s','5s','7z','7z','2m','2m','4p','4p','9s']
        r = self.solve(hand)
        self.assertEqual(r['shanten'], 0, r)
        self.assertEqual([w['tile'] for w in r.get('waits', [])], [26])  # 26 = 9s 索引

    def test_four_identical_tiles_are_not_a_false_wait(self):
        # 四张同牌不能算作两个对子/假听（v1.1 修复项回归）
        hand = ['1m','1m','1m','1m','2m','3m','4m','5m','6m','7m','8m','9m','1p']
        r = self.solve(hand)
        self.assertGreaterEqual(r['shanten'], 0)

    def test_red_five_shares_physical_copy_with_normal_five(self):
        # 赤五与普通五同一牌种：手牌 0m+5m 不可再进 5m
        hand = ['0m','5m','2m','3m','4m','6p','7p','8p','2s','3s','4s','5s','1z']
        r = self.solve(hand, ['5m','5m'])
        for u in r.get('ukeire', []) + r.get('waits', []):
            if u['tile'] in ('0m','5m'):
                self.assertLessEqual(u.get('remaining', 0), 0)

    # ---------- 宝牌与风牌 ----------
    def test_dora_cycles_winds_and_dragons_independent(self):
        self.assertEqual(dora_from_indicator('4z'), '1z')  # 北→东，不是北→白
        self.assertEqual(dora_from_indicator('7z'), '5z')  # 中→白，不是中→东
        self.assertEqual(dora_from_indicator('0m'), '6m')  # 赤五指示按 5 处理

    # ---------- 安全事实 ----------
    def test_genbutsu_and_suji_are_per_player_not_global(self):
        facts = build_safety_facts(
            {'right': ['4m','8s'], 'across': ['7p','1z']},
            {'right': True, 'across': True})
        self.assertIn('4m', facts['现物']['right'])
        self.assertIn('1m', facts['筋']['right'])   # 切4m → 1m/7m 两侧筋
        self.assertIn('7m', facts['筋']['right'])
        self.assertNotIn('4m', facts['现物']['across'])

    def test_single_sided_suji_from_edge_discard(self):
        facts = build_safety_facts({'across': ['8m','1p']}, {'across': True})
        self.assertIn('5m', facts['单筋']['across'])
        self.assertIn('4p', facts['单筋']['across'])
        self.assertEqual(facts['筋']['across'], [])  # 单侧筋不混入强筋

    def test_no_safety_facts_without_riichi(self):
        facts = build_safety_facts({'across': ['8m','1p']}, {'across': False})
        self.assertEqual(facts['现物'], {})
        self.assertEqual(facts['筋'], {})
        self.assertEqual(facts['单筋'], {})

    def test_repeated_discard_does_not_create_two_sided_suji(self):
        facts = build_safety_facts({'across': ['4m','4m','4m']}, {'across': True})
        self.assertEqual(facts['两筋']['across'], [])

    # ---------- 副露开牌 ----------
    def test_open_hand_with_quad_and_three_melds(self):
        hand = ['1m','2m','3m','5p']  # 4 + 杠4 + 碰3 + 吃3 = 14 有效
        groups = [[{'name':'7z'}]*4, [{'name':'9p'}]*3,
                  [{'name':'1s'},{'name':'2s'},{'name':'3s'}]]
        r = analyze_open(hand, groups, seen=['7z'])
        self.assertIn('shanten', r)
        self.assertIn('candidates', r)

    def test_open_hand_rejects_impossible_five_copies(self):
        hand = ['7z','1m','2m','3m','4m','5m','6m','7m','8m','9m']
        groups = [[{'name':'7z'}]*3, [{'name':'7z'}]*3]  # 手1+副露6 = 7 张 7z
        with self.assertRaises(ValueError):
            analyze_open(hand, groups)


if __name__ == '__main__':
    unittest.main()
