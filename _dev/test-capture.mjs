/**
 * 截图识别管线回归测试（不需要真屏幕）
 * 用 _dev/mock 里渲染好的两张"假客户端"截图跑：
 *   1. 无模板 -> 聚类（应得 10 类）
 *   2. 人工标注 -> 学模板
 *   3. 用偏了 6 像素的框重新识别（应全对）
 *   4. 换位置/换尺寸/换手牌的第二张图（应全对）
 *
 * 用法: node _dev\test-capture.mjs
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { detectPython, PythonWorker } from '../server.mjs';

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const PY = detectPython();
if (!PY) throw new Error('需要安装 Pillow/numpy 的 Python；请设置 PYTHON');

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'mjcap-'));
const env = { ...process.env, MJ_TPL_DIR: path.join(tmp, 'tpl'), MJ_CELL_DIR: path.join(tmp, 'cells'), MJ_CALIB: path.join(tmp, 'calib.json') };

let pass = 0, fail = 0;
const bad = [];
function ok(cond, label, extra) {
  if (cond) pass++;
  else { fail++; bad.push(label + (extra ? ' → ' + JSON.stringify(extra) : '')); }
}

function py(payload) {
  const r = spawnSync(PY, ['-B', path.join(DIR, 'capture.py'), JSON.stringify(payload)], { env, encoding: 'utf8', windowsHide: true, timeout: 20000 });
  if (r.error || r.status !== 0) throw new Error('Python 识别进程失败: ' + (r.error?.message || r.stderr));
  const line = (r.stdout || '').trim().split('\n').pop();
  try { return JSON.parse(line); }
  catch { throw new Error('python 输出无法解析: ' + (r.stdout || '') + (r.stderr || '').slice(0, 300)); }
}

try {
const mockA = path.join(DIR, '_dev', 'mock', 'screenA.png');
const mockB = path.join(DIR, '_dev', 'mock', 'screenA2.png');
const BOX_A = [280, 780, 1120, 110];
const BOX_A_OFF = [274, 774, 1132, 112];   // 故意偏 6 像素 + 放大

/* 1. 无模板 -> 聚类 */
let r = py({ mode: 'scan', image: mockA, box: BOX_A, n: 14 });
ok(r.ok, '1.0 扫描成功', r.error);
ok(r.clusters.length === 10, '1.1 无模板时聚成 10 类', r.clusters.length);
ok(r.unknown === 14, '1.2 全部未识别', r.unknown);

/* 2. 按格子标注 -> 学模板 */
const MAP = { 0: 0, 2: 1, 4: 2, 6: 12, 7: 13, 8: 14, 9: 24, 10: 25, 11: 26, 12: 33, 13: 33 };
const lr = py({ mode: 'learn', assign: MAP, scanId: r.scanId });
ok(lr.ok && lr.saved === 11, '2.1 学会 11 个格子', lr);

/* 3. 偏 6 像素重新识别 */
r = py({ mode: 'scan', image: mockA, box: BOX_A_OFF, n: 14 });
ok(r.hand === '112233m456p789s77z', '3.1 偏 6px 仍全对', { hand: r.hand, unknown: r.unknown });
ok(r.unknown === 0, '3.2 没有未识别', r.unknown);
const minScore = Math.min(...r.cells.map((c) => c.score));
const minMargin = Math.min(...r.cells.map((c) => c.margin));
ok(minScore > 0.85, '3.3 最低相关度 > 0.85', minScore);
ok(minMargin > 0.03, '3.4 最低区分度 > 0.03', minMargin);

/* 4. 换位置/换尺寸/换手牌 */
r = py({ mode: 'scan', image: mockB, box: [200, 700, 1400, 130], n: 14 });
ok(r.hand === '112233m445566p77s', '4.1 换几何+新牌仍全对', { hand: r.hand, unknown: r.unknown });

/* 5. 速度 */
const t0 = Date.now();
py({ mode: 'scan', image: mockA, box: BOX_A, n: 14 });
const ms = Date.now() - t0;
ok(ms < 3000, '5.1 单次识别 < 3s', ms);

console.log(`\n通过 ${pass} / ${pass + fail}   单次识别 ${ms} ms（含 python 启动）`);
if (fail) {
  console.log('失败项：');
  bad.forEach((b) => console.log('  ✗ ' + b));
  throw new Error('截图回归测试失败');
}
console.log('全部通过 ✅');
const worker = new PythonWorker(PY, { spawnImpl: (executable, args, options) => spawn(executable, args, { ...options, env }) });
try {
  const first = await worker.run({ mode: 'scan', image: mockA, box: BOX_A, n: 14 });
  assert.equal(first.ok, true);
  const [second, stale] = await Promise.all([
    worker.run({ mode: 'scan', image: mockB, box: [200, 700, 1400, 130], n: 14 }),
    worker.run({ mode: 'learn', scanId: first.scanId, assign: { 0: 0 }, n: 14 }),
  ]);
  assert.equal(second.ok, true);
  assert.equal(second.hand, '112233m445566p77s');
  assert.equal(stale.ok, false);
  assert.match(stale.error, /过期/);
  const learned = await worker.run({ mode: 'learn', scanId: second.scanId, assign: { 0: 0 }, n: 14 });
  assert.equal(learned.saved, 1);
  console.log('常驻进程：并发扫描与过期学习隔离通过 ✅');
} finally { worker.close(); }
const validation = spawnSync(PY, ['-B', path.join(DIR, '_dev', 'test-capture-validation.py')], {
  env, encoding: 'utf8', windowsHide: true, timeout: 30000,
});
process.stdout.write(validation.stdout || '');
process.stderr.write(validation.stderr || '');
if (validation.error || validation.status !== 0) throw new Error('截图输入/会话校验测试失败: ' + (validation.error?.message || validation.status));
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
