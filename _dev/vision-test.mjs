/* eslint-disable no-console */
/**
 * 读图能力测试：把牌面截图丢给本地模型，看它能不能读成 m/p/s/z 记法
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const KEY = process.env.SPARK_KEY || '';
const URL = process.env.SPARK_URL || '';
const MODEL = process.env.SPARK_MODEL || '';
if (!URL || !MODEL) throw new Error('请先设置 SPARK_URL 和 SPARK_MODEL；SPARK_KEY 按服务要求填写。');

const HTML = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '雀魂友人场助手.html');
const core = fs.readFileSync(HTML, 'utf8').match(/<script id="mj-core">([\s\S]*?)<\/script>/)[1];
const MJ = new Function(core + '\nreturn MJ;')();

const PROMPT = '这张图是一手日麻（立直麻雀）牌。请从左到右读出所有牌，用 m=万 p=筒 s=索 z=字 的记法输出，'
  + '例如 123m456p789s11z。只输出这一串，不要任何解释。';

function expand(str) {
  try {
    const c = MJ.parseTiles(str);
    return MJ.toText(c);
  } catch { return null; }
}

async function ask(imagePath) {
  const b64 = fs.readFileSync(imagePath).toString('base64');
  const body = {
    model: MODEL,
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: PROMPT },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,' + b64 } },
      ],
    }],
    max_tokens: 96,
    stream: false,
    reasoning_effort: 'none',
  };
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
    ms, text: (msg.content || '').trim(), think: (msg.reasoning_content || '').trim(),
    usage: j.usage, timings: j.timings,
  };
}

const cases = [
  { file: 'handA.png', truth: '123456789m2345p4z', desc: '文字风格牌面 14 张（9 万 + 4 筒 + 北）' },
  { file: 'handB.png', truth: '234m567p123s999s33z', desc: 'Unicode 真实麻将字形 14 张' },
];

(async () => {
  for (const c of cases) {
    const p = path.join(path.dirname(fileURLToPath(import.meta.url)), 'vision', c.file);
    console.log('\n' + '─'.repeat(70));
    console.log(`${c.file}  ${c.desc}`);
    console.log(`答案应该是: ${c.truth}`);
    try {
      const r = await ask(p);
      const gotRaw = r.text;
      const m = gotRaw.match(/[0-9mpsz]+/gi);
      const got = m ? expand(m.join('').toLowerCase()) : null;
      const exp = expand(c.truth);
      const ok = got === exp;
      console.log(`模型输出: ${JSON.stringify(gotRaw.slice(0, 80))}`);
      console.log(`解析成  : ${got || '(无法解析)'}`);
      console.log(`结果    : ${ok ? '✅ 完全正确' : '❌ 读错'}`);
      console.log(`耗时    : ${(r.ms / 1000).toFixed(2)}s   in=${r.usage?.prompt_tokens} `
        + `out=${r.usage?.completion_tokens}   图像token≈${r.usage?.prompt_tokens_details?.image_tokens ?? '?'}`);
      console.log(`timings : prefill ${(r.timings?.prompt_ms / 1000).toFixed(2)}s / gen ${(r.timings?.predicted_ms / 1000).toFixed(2)}s`);
      if (!ok && got) {
        // 找出差异
        const gc = MJ.parseTiles(got), ec = MJ.parseTiles(exp);
        const diff = [];
        for (let i = 0; i < 34; i++) if (gc[i] !== ec[i]) diff.push(`${MJ.NAMES[i]} 期望${ec[i]} 读成${gc[i]}`);
        console.log(`差异    : ${diff.join('; ')}`);
      }
    } catch (e) {
      console.log('ERROR ' + e.message);
    }
  }
})();
