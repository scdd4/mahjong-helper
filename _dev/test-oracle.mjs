// 独立参考：枚举全字牌的所有合法目标和型，直接求最少补牌数。
// 不调用待测引擎的和牌、拆分或向听辅助函数生成期望值。
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(root, '雀魂友人场助手.html'), 'utf8');
const core = html.match(/<script id="mj-core">([\s\S]*?)<\/script>/)[1];
const MJ = new Function(core + '\nreturn MJ;')();
const targets = [];
for (let a = 0; a < 7; a++) for (let b = a + 1; b < 7; b++) {
  for (let c = b + 1; c < 7; c++) for (let d = c + 1; d < 7; d++) {
    for (let pair = 0; pair < 7; pair++) {
      if ([a, b, c, d].includes(pair)) continue;
      const target = Array(7).fill(0);
      for (const tile of [a, b, c, d]) target[tile] = 3;
      target[pair] = 2;
      targets.push(target);
    }
  }
}
targets.push(Array(7).fill(2)); // 七对子

function reference(honors) {
  let overlap = 0;
  for (const target of targets) {
    overlap = Math.max(overlap, honors.reduce((n, v, i) => n + Math.min(v, target[i]), 0));
  }
  // 国士的六种数牌幺九均缺；只需枚举哪一种字牌作对子。
  for (let pair = 0; pair < 7; pair++) {
    overlap = Math.max(overlap, honors.reduce((n, v, i) => n + Math.min(v, i === pair ? 2 : 1), 0));
  }
  return 13 - overlap;
}
let seed = 0x145359;
function random() {
  seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
  return (seed >>> 0) / 4294967296;
}
const examples = [[4, 3, 3, 3, 0, 0, 0], [4, 4, 4, 1, 0, 0, 0]];
for (let k = 0; k < 400; k++) {
  const h = Array(7).fill(0);
  for (let n = 13 + (k % 2); n > 0;) {
    const i = Math.floor(random() * 7);
    if (h[i] < 4) { h[i]++; n--; }
  }
  examples.push(h);
}
const t0 = performance.now();
for (const h of examples) {
  const counts = Array(27).fill(0).concat(h);
  assert.equal(MJ.bestShanten(counts).value, reference(h), `全字牌计数 ${h}`);
}
console.log(`独立目标枚举：${examples.length}/${examples.length} 通过（${(performance.now() - t0).toFixed(0)} ms）`);
