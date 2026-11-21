"""Jev-like 决策契约测试：全部用进程内假服务，不连真实模型。"""
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desktop_runtime import Settings
from spark_strategy import SparkClient
from jev_decision import (TOKEN_BASE, TOKEN_FIRST_ID, MAX_CHOICES, candidates_from_calculations,
                          build_prompt, parse_logprobs, confidence_level, build_report)

HAND = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m', '2p', '3p', '4p', '5p', '4z']
CALC = {'shanten': 0, 'candidates': [
    {'tile': '4z', 'shanten': 0, 'ukeire': 40, 'kinds': 6},
    {'tile': '9m', 'shanten': 0, 'ukeire': 38, 'kinds': 5},
    {'tile': '5p', 'shanten': 1, 'ukeire': 30, 'kinds': 4}], 'waits': [], 'ukeire': []}
EV = {'hand': {'reliable': True, 'tiles': [{'name': t} for t in HAND]},
      'rivers': {'right': [{'name': '1z'}]}, 'riichi': {'right': True},
      'dora_indicators': ['5z'], 'melds': {}, 'warnings': []}


class JevDecisionTests(unittest.TestCase):
    def test_token_ids_are_contiguous_a_to_n(self):
        self.assertEqual(TOKEN_FIRST_ID, 32)
        self.assertEqual(ord('A'), TOKEN_BASE)
        for i in range(MAX_CHOICES):
            self.assertEqual(ord(chr(TOKEN_BASE + i)) - TOKEN_BASE, i)

    def test_candidates_are_playable_and_bounded(self):
        c = candidates_from_calculations(CALC, HAND, EV)
        self.assertLessEqual(len(c), MAX_CHOICES)
        self.assertEqual(c[0], '4z')
        self.assertTrue(all(tile in HAND for tile in c))
        self.assertEqual(len(set(c)), len(c))

    def test_build_prompt_maps_letters_to_tiles(self):
        c = candidates_from_calculations(CALC, HAND, EV)
        text = build_prompt(HAND, CALC, EV, c)
        self.assertIn('A=4z', text)
        self.assertIn('B=9m', text)
        self.assertIn('立直家：right', text)
        self.assertIn('只输出一个字母', text)

    def test_parse_logprobs_normalises_and_ranks(self):
        c = ['4z', '9m', '5p', '1m']
        choice = {'logprobs': {'content': [{'token': 'A', 'top_logprobs': [
            {'token': 'A', 'logprob': -0.1}, {'token': 'B', 'logprob': -2.3},
            {'token': 'C', 'logprob': -4.0}]}]}}
        tile, conf, probs = parse_logprobs(choice, c)
        self.assertEqual(tile, '4z')
        # exp(-0.1)+exp(-2.3)+exp(-4) 归一化；未列出的 D 得极小值
        self.assertAlmostEqual(conf, 0.884, delta=0.01)
        self.assertAlmostEqual(sum(probs.values()), 1.0, delta=1e-6)
        self.assertGreater(probs['4z'], probs['9m'])
        self.assertLess(probs['1m'], 1e-6)

    def test_confidence_levels(self):
        self.assertEqual(confidence_level(0.95), 'high')
        self.assertEqual(confidence_level(0.45), 'medium')
        self.assertEqual(confidence_level(0.12), 'low')

    def test_report_is_concise_and_carries_probabilities(self):
        probs = {'4z': 0.9, '9m': 0.08, '5p': 0.02}
        r = build_report('4z', 0.9, probs, ['4z', '9m', '5p'], HAND, EV,
                         '本地核验', [], 2.0, 'test-model', True)
        self.assertEqual(r['mode'], 'jev')
        self.assertEqual(r['report']['strategy']['recommendation']['tile'], '4z')
        self.assertEqual(r['report']['strategy']['stance'], 'Jev决策')
        self.assertIn('概率', r['report']['strategy']['recommendation']['reason'])
        self.assertEqual(r['jev_confidence'], 0.9)
        self.assertTrue(r['jev_reasoning_used'])

    def test_decide_jev_uses_constrained_tokens_and_logprobs(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append(body)
                # 假服务：返回候选 A 高概率
                reply = {'model': body['model'], 'choices': [{'index': 0, 'message': {
                    'role': 'assistant', 'content': 'A', 'reasoning_content': 'thinking...'},
                    'finish_reason': 'stop', 'logprobs': {'content': [{'token': 'A', 'top_logprobs': [
                        {'token': 'A', 'logprob': -0.05}, {'token': 'B', 'logprob': -3.0},
                        {'token': 'C', 'logprob': -5.0}]}]}}], 'usage': {}}
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            settings = Settings(base_url='http://127.0.0.1:%d/v1' % server.server_port,
                                api_key='test-key', analysis_mode='jev')
            from PIL import Image
            with Image.new('RGB', (32, 32)) as image:
                result = SparkClient(settings).decide_jev(image, EV, CALC)
            self.assertEqual(len(received), 1)
            req = received[0]
            self.assertTrue(req['logprobs'])
            self.assertFalse(req['stream'])
            self.assertEqual(req['allowed_token_ids'][0], TOKEN_FIRST_ID)
            self.assertEqual(len(req['allowed_token_ids']), len(candidates_from_calculations(CALC, HAND, EV)))
            self.assertFalse(req['chat_template_kwargs']['enable_thinking'])  # System One：不需要 CoT
            self.assertEqual(result['report']['strategy']['recommendation']['tile'], '4z')
            self.assertEqual(result['mode'], 'jev')
            self.assertEqual(result['jev_level'], 'high')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_settings_accepts_jev_mode(self):
        self.assertEqual(Settings(analysis_mode='jev').validate().analysis_mode, 'jev')

    def test_jev_mode_rejects_invalid_mode(self):
        with self.assertRaises(ValueError):
            Settings(analysis_mode='jev-like').validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
