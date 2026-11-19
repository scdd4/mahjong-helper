/* eslint-disable no-console */
/**
 * 延迟探针：把本地模型的响应时间压到极限，看下限在哪
 */
const KEY = process.env.SPARK_KEY || '';
const URL = process.env.SPARK_URL || '';
const MODEL = process.env.SPARK_MODEL || '';
if (!URL || !MODEL) throw new Error('请先设置 SPARK_URL 和 SPARK_MODEL；SPARK_KEY 按服务要求填写。');

async function chat(messages, opts = {}) {
  const body = {
    model: MODEL,
    messages,
    max_tokens: opts.maxTokens ?? 32,
    stream: false,
  };
  if (opts.effort) body.reasoning_effort = opts.effort;
  if (opts.prefix) body.strata_prefix = opts.prefix;
  if (opts.temp != null) body.temperature = opts.temp;
  const t0 = performance.now();
  const res = await fetch(URL, {
    method: 'POST',
    signal: AbortSignal.timeout(20000),
    headers: { Authorization: 'Bearer ' + KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const ms = performance.now() - t0;
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const j = await res.json();
  const m = j.choices[0].message;
  return {
    ms,
    text: (m.content || '').trim(),
    think: (m.reasoning_content || '').trim(),
    tin: j.usage.prompt_tokens,
    tout: j.usage.completion_tokens,
    cached: j.usage.prompt_tokens_details?.cached_tokens ?? 0,
    pre: Math.round(j.timings?.prompt_ms ?? 0),
    gen: Math.round(j.timings?.predicted_ms ?? 0),
  };
}

const show = (tag, r) =>
  console.log(`${tag.padEnd(26)} total ${(r.ms / 1000).toFixed(2)}s  `
    + `(prefill ${(r.pre / 1000).toFixed(2)}s + gen ${(r.gen / 1000).toFixed(2)}s)  `
    + `in=${r.tin} cached=${r.cached} out=${r.tout}  → ${JSON.stringify(r.text.slice(0, 30))}`);

(async () => {
  const HAND = '123456789m2345p4z';       // 正解：打 4z（北）
  const HAND2 = '246m2577p1777s113z';     // 正解：打 3z（西）

  console.log('\n===== 1. 极简英文提示，关思考 =====');
  const sysEn = 'Riichi mahjong. Reply with only the tile to discard, like 4z.';
  for (let i = 0; i < 3; i++) {
    const r = await chat([
      { role: 'system', content: sysEn },
      { role: 'user', content: `hand=${HAND} d=` },
    ], { effort: 'none', maxTokens: 16 });
    show(`en-min #${i + 1}`, r);
    console.log(`    answer=${JSON.stringify(r.text)}`);
  }

  console.log('\n===== 2. 同样提示 + strata_prefix 命中缓存 =====');
  for (let i = 0; i < 3; i++) {
    const r = await chat([
      { role: 'system', content: sysEn },
      { role: 'user', content: `hand=${HAND2} d=` },
    ], { effort: 'none', maxTokens: 16, prefix: { messages: 1 } });
    show(`prefix #${i + 1}`, r);
    console.log(`    answer=${JSON.stringify(r.text)}`);
  }

  console.log('\n===== 3. 中文极简，关思考（对比分词代价） =====');
  for (let i = 0; i < 2; i++) {
    const r = await chat([
      { role: 'system', content: '日麻。只回答要打的牌，如 4z。' },
      { role: 'user', content: `手牌=${HAND} 打` },
    ], { effort: 'none', maxTokens: 16 });
    show(`zh-min #${i + 1}`, r);
    console.log(`    answer=${JSON.stringify(r.text)}`);
  }

  console.log('\n===== 4. 让模型「解释求解器的结论」（推荐用法） =====');
  const explainPrompt = '求解器结论：打 4z(北)，剩余进张 18 种 64 张；次优 9m 只有 56 张。'
    + '请用不超过 30 个字解释为什么打北（面向日麻新手）。';
  for (let i = 0; i < 2; i++) {
    const r = await chat([{ role: 'user', content: explainPrompt }], { effort: 'none', maxTokens: 80 });
    show(`explain #${i + 1}`, r);
    console.log(`    ${r.text}`);
  }

  console.log('\n===== 5. 关思考 vs low 的正确率/耗时（同一手牌） =====');
  const hard = '1369m2569p77899s3z';   // 正解：西(3z)
  for (const eff of ['none', 'low', 'medium']) {
    const r = await chat([
      { role: 'system', content: '你是日麻牌手。只输出要打的牌，格式如 3z。' },
      { role: 'user', content: `东场东家，门清13张（等摸牌）14张：${hard}` },
    ], { effort: eff, maxTokens: 400 });
    show(`effort=${eff}`, r);
    console.log(`    think=${JSON.stringify(r.think.slice(0, 60))} ans=${JSON.stringify(r.text)}`);
  }
})();
