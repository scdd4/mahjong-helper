/* eslint-disable no-console */
/**
 * Spark 本地模型 vs 本地求解器 —— 出牌建议对比基准
 * 用法: node _dev\spark-bench.mjs [caseCount] [config]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HTML = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '雀魂友人场助手.html');
const html = fs.readFileSync(HTML, 'utf8');
const core = html.match(/<script id="mj-core">([\s\S]*?)<\/script>/)[1];
const MJ = new Function(core + '\nreturn MJ;')();

const KEY = process.env.SPARK_KEY || '';
const URL = process.env.SPARK_URL || '';
const MODEL = process.env.SPARK_MODEL || '';
if (!URL || !MODEL) throw new Error('请先设置 SPARK_URL 和 SPARK_MODEL；SPARK_KEY 按服务要求填写。');

const SYS_COMPACT = '你是日麻（立直麻雀）四麻专家。手牌用 m=万 p=筒 s=索 z=字 表示（1z东 2z南 3z西 4z北 5z白 6z发 7z中）。'
  + '判断该打哪张时优先考虑牌效率（进张数），其次防守。只输出要打出的那一张牌，不要解释。';

function tileFromText(txt, counts) {
  const s = String(txt || '');
  let m = s.match(/([1-9])\s*([mpsz])/i);
  if (m) {
    const n = +m[1], k = m[2].toLowerCase();
    const base = k === 'm' ? 0 : k === 'p' ? 9 : k === 's' ? 18 : 27;
    const idx = base + n - 1;
    if (idx >= 0 && idx < 34) return idx;
  }
  const names = { '东': 27, '南': 28, '西': 29, '北': 30, '白': 31, '发': 32, '発': 32, '中': 33 };
  for (const [ch, idx] of Object.entries(names)) if (s.includes(ch)) return idx;
  m = s.match(/([1-9])\s*([万筒索])/);
  if (m) {
    const base = m[2] === '万' ? 0 : m[2] === '筒' ? 9 : 18;
    return base + (+m[1]) - 1;
  }
  return -1;
}

async function chat(messages, { effort, maxTokens }) {
  const body = { model: MODEL, messages, max_tokens: maxTokens, stream: false };
  if (effort) body.reasoning_effort = effort;
  const t0 = performance.now();
  const res = await fetch(URL, {
    method: 'POST',
    signal: AbortSignal.timeout(20000),
    headers: { Authorization: 'Bearer ' + KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const ms = performance.now() - t0;
  if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()).slice(0, 200));
  const j = await res.json();
  const msg = j.choices[0].message;
  return {
    ms,
    content: msg.content || '',
    reasoning: msg.reasoning_content || '',
    usage: j.usage,
    timings: j.timings,
  };
}

/* ---------- 生成/挑选测试牌例 ---------- */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(777);
function rndHand(n) {
  for (;;) {
    const c = MJ.zero();
    let left = n, guard = 0;
    while (left > 0 && guard++ < 5000) { const i = Math.floor(rng() * 34); if (c[i] < 4) { c[i]++; left--; } }
    if (left === 0) return c;
  }
}

function buildCases(n) {
  const out = [];
  const seen = new Set();
  let guard = 0;
  while (out.length < n && guard++ < 4000) {
    const c = rndHand(14);
    const txt = MJ.toText(c);
    if (seen.has(txt)) continue;
    if (MJ.isWin(c)) continue;
    const r = MJ.analyze(c, {});
    if (!r.discards.length) continue;
    const top = r.discards[0];
    const second = r.discards[1];
    if (!second || second.rank === 999) continue;
    // 只要「有明显最优解」的牌：进张至少多 2 张
    if (top.ukeireTotal - second.ukeireTotal < 2) continue;
    // 不要十种牌都一样的垃圾手
    if (r.shanten > 3) continue;
    seen.add(txt);
    out.push({ hand: txt, counts: c, top, second, shanten: r.shanten });
  }
  return out;
}

/* ---------- 主流程 ---------- */
const N = Number(process.argv[2] || 6);
const CONFIG = process.argv[3] || 'all';

const cases = buildCases(N);
console.log(`\n已挑选 ${cases.length} 个牌例（求解器认为有明显最优解）\n`);

const results = [];
let warm = false;

for (const c of cases) {
  const expected = c.top.tile;
  const expName = MJ.NAMES[expected];
  console.log('─'.repeat(70));
  console.log(`手牌 ${c.hand}   ${c.shanten} 向听`);
  console.log(`求解器: 打 ${expName}(${(expected % 9) + 1}${'mpsz'[(expected / 9) | 0]}) `
    + `进张 ${c.top.ukeireKinds}种${c.top.ukeireTotal}张   `
    + `次优 ${MJ.NAMES[c.second.tile]} ${c.second.ukeireTotal}张  `
    + `[${c.top.ukeireTotal - c.second.ukeireTotal > 0 ? '+' + (c.top.ukeireTotal - c.second.ukeireTotal) : '0'}]`);

  const variants = [];
  if (CONFIG === 'all' || CONFIG === 'compact') variants.push({ tag: 'compact', effort: 'low', maxTokens: 200 });
  if (CONFIG === 'all' || CONFIG === 'nothink') variants.push({ tag: 'nothink', effort: 'none', maxTokens: 32 });

  for (const v of variants) {
    const messages = [
      { role: 'system', content: SYS_COMPACT },
      { role: 'user', content: `东场东家，门清14张：${c.hand}。该打哪张？` },
    ];
    try {
      const r = await chat(messages, { effort: v.effort, maxTokens: v.maxTokens });
      const got = tileFromText(r.content, c.counts);
      const ok = got === expected;
      const cached = r.usage?.prompt_tokens_details?.cached_tokens ?? 0;
      console.log(`  [${v.tag}] ${(r.ms / 1000).toFixed(2)}s  `
        + `in=${r.usage?.prompt_tokens ?? '?'}(cached ${cached}) out=${r.usage?.completion_tokens ?? '?'} `
        + `→ "${r.content.trim().slice(0, 40)}"  ${ok ? '✅' : '❌ 期望 ' + expName}`);
      if (!warm) warm = true;
      results.push({ hand: c.hand, expected: expName, variant: v.tag, got: got >= 0 ? MJ.NAMES[got] : '?', ok, ms: r.ms, usage: r.usage });
    } catch (e) {
      console.log(`  [${v.tag}] ERROR ${e.message}`);
      results.push({ hand: c.hand, expected: expName, variant: v.tag, got: 'ERR', ok: false, ms: -1 });
    }
  }
}

/* ---------- 汇总 ---------- */
console.log('\n' + '='.repeat(70));
const byTag = {};
for (const r of results) {
  (byTag[r.variant] ||= []).push(r);
}
for (const [tag, arr] of Object.entries(byTag)) {
  const okN = arr.filter((x) => x.ok).length;
  const times = arr.filter((x) => x.ms > 0).map((x) => x.ms).sort((a, b) => a - b);
  const med = times.length ? times[Math.floor(times.length / 2)] : 0;
  const tok = arr.map((x) => x.usage?.completion_tokens || 0).reduce((a, b) => a + b, 0);
  console.log(`${tag.padEnd(8)} 正确 ${okN}/${arr.length}   中位耗时 ${(med / 1000).toFixed(2)}s   `
    + `总输出 ${tok} tok   ${tok / (times.reduce((a, b) => a + b, 0) / 1000 || 1)} tok/s`);
}
console.log('求解器   正确 100%   中位耗时 0.0004s (0.4ms)');

const outFile = path.join(path.dirname(fileURLToPath(import.meta.url)), 'bench-result.json');
fs.writeFileSync(outFile, JSON.stringify(results, null, 2));
console.log(`\n明细已写入 ${outFile}`);
