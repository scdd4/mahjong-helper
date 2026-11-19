/** One-button page state regressions. No desktop capture or running server. */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const DIR = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(DIR, '..', '一键分析.html'), 'utf8');
const source = html.match(/<script id="auto-ui">([\s\S]*?)<\/script>/)[1];

function harness() {
  const nodes = new Map(), requests = [], timers = new Map(), windowEvents = new Map();
  let timerId = 0, now = 0;
  function el(id) {
    if (!nodes.has(id)) nodes.set(id, {
      textContent: '', className: '', hidden: ['result', 'waits'].includes(id), disabled: false, attributes: {}, events: {},
      setAttribute(key, value) { this.attributes[key] = value; },
      addEventListener(type, fn) { this.events[type] = fn; },
      click() { return this.events.click?.(); },
    });
    return nodes.get(id);
  }
  const context = vm.createContext({
    document: { getElementById: el },
    window: { addEventListener(type, fn) { windowEvents.set(type, fn); } },
    AbortController,
    setTimeout(fn, delay) { const id = ++timerId; timers.set(id, { fn, at: now + delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
    fetch(url, options) {
      return new Promise((resolve, reject) => requests.push({
        url, options, reject,
        resolve(data, ok = true) { resolve({ ok, json: async () => data }); },
        invalidJson() { resolve({ ok: true, json: async () => { throw new Error('invalid json'); } }); },
      }));
    },
  });
  vm.runInContext(source, context);
  return {
    el, requests, context,
    pagehide() { windowEvents.get('pagehide')(); },
    tick(ms) {
      const end = now + ms;
      for (;;) {
        const next = [...timers].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!next) break;
        now = next[1].at; timers.delete(next[0]); next[1].fn();
      }
      now = end;
    },
  };
}
function success(overrides = {}) {
  return {
    ok: true, hand: '123456789m2345p4z', tiles: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 30],
    window: { title: '雀魂', width: 1280, height: 720 }, ms: 350,
    analysis: { total: 14, shanten: 0, win: false, tenpai: false,
      best: { tile: '北', index: 30, shanten: 0, ukeire: 6, kinds: 2, tenpai: true },
      waits: [{ tile: '2筒', left: 3 }, { tile: '5筒', left: 3 }] },
    ...overrides,
  };
}
async function flush() { for (let i = 0; i < 12; i++) await Promise.resolve(); }
let passed = 0;
async function test(name, fn) { await fn(); passed++; console.log('OK ' + name); }

await test('the page contains exactly one button and no input or calibration controls', () => {
  assert.equal((html.match(/<button\b/g) || []).length, 1);
  assert.doesNotMatch(html, /<(input|select|textarea)\b/i);
  assert.doesNotMatch(source, /\/api\/(capture|learn|explain)/);
});
await test('opening the page makes no network request', () => {
  const h = harness(); h.tick(60000); assert.equal(h.requests.length, 0); assert.equal(h.el('result').hidden, true);
});
await test('one click sends only the agreed request and blocks repeated clicks while busy', () => {
  const h = harness(); h.el('analyze').click(); h.el('analyze').click();
  assert.equal(h.requests.length, 1); assert.equal(h.requests[0].url, '/api/analyze-window');
  assert.equal(h.requests[0].options.method, 'POST'); assert.equal(h.requests[0].options.body, '{}');
  assert.equal(h.el('analyze').disabled, true); assert.equal(h.el('analyze').attributes['aria-busy'], 'true');
});
await test('a complete response displays the discard, waits, and recognized hand without confirmation', async () => {
  const h = harness(); h.el('analyze').click(); h.requests[0].resolve(success()); await flush();
  assert.equal(h.el('result').hidden, false); assert.equal(h.el('recommendation').textContent, '建议打 北');
  assert.match(h.el('detail').textContent, /2 种.*6 张/); assert.match(h.el('waits').textContent, /2筒.*5筒/);
  assert.match(h.el('hand').textContent, /1万.*北/); assert.equal(h.el('analyze').disabled, false);
  assert.match(h.el('status').textContent, /雀魂.*0.35 秒/);
});
await test('13-tile hands describe waits without inventing a discard', async () => {
  const h = harness(); h.el('analyze').click(); const data = success(); data.tiles.pop(); data.analysis.total = 13; data.analysis.tenpai = true; data.analysis.best = null;
  h.requests[0].resolve(data); await flush(); assert.equal(h.el('recommendation').textContent, '已经听牌'); assert.match(h.el('detail').textContent, /等待摸牌/);
});

await test('red-five identity and the physical discard position are visible', async () => {
  const h=harness();h.el('analyze').click();const data=success();
  data.tileCodes=['1m','2m','3m','4m','5m','6m','7m','8m','9m','2p','3p','4p','0p','4z'];
  data.analysis.best={tile:'5筒',index:13,red:true,position:13,shanten:0,ukeire:3,kinds:1,tenpai:true};
  h.requests[0].resolve(data);await flush();
  assert.equal(h.el('recommendation').textContent,'建议打 赤5筒');
  assert.match(h.el('detail').textContent,/第 13 张/);assert.match(h.el('hand').textContent,/赤5筒/);
});
await test('a completed winning shape is labelled without claiming a legal win', async () => {
  const h = harness(); h.el('analyze').click(); const data = success(); data.analysis.win = true; data.analysis.shanten = -1; data.analysis.best = null;
  h.requests[0].resolve(data); await flush(); assert.equal(h.el('recommendation').textContent, '和牌形已完成'); assert.match(h.el('detail').textContent, /役种和振听/); assert.equal(h.el('waits').hidden, true);
});
await test('starting another scan immediately clears the previous suggestion and hand', async () => {
  const h = harness(); h.el('analyze').click(); h.requests[0].resolve(success()); await flush(); h.el('analyze').click();
  assert.equal(h.el('result').hidden, true); assert.equal(h.el('recommendation').textContent, ''); assert.equal(h.el('hand').textContent, '');
});
await test('API errors show a retry explanation and cannot retain the previous suggestion', async () => {
  const h = harness(); h.el('analyze').click(); h.requests[0].resolve(success()); await flush(); h.el('analyze').click();
  h.requests[1].resolve({ ok: false, error: '未找到雀魂窗口', code: 'WINDOW_NOT_FOUND' }, false); await flush();
  assert.equal(h.el('result').hidden, true); assert.match(h.el('status').textContent, /未找到雀魂窗口.*重试/); assert.equal(h.el('analyze').disabled, false);
});
await test('HTTP failures are rejected even if their JSON claims success', async () => {
  const h = harness(); h.el('analyze').click(); h.requests[0].resolve(success(), false); await flush();
  assert.equal(h.el('result').hidden, true); assert.match(h.el('status').className, /error/);
});
await test('invalid JSON produces a useful service error', async () => {
  const h = harness(); h.el('analyze').click(); h.requests[0].invalidJson(); await flush();
  assert.equal(h.el('analyze').disabled, false); assert.match(h.el('status').textContent, /本地服务返回异常.*重试/);
});
await test('network failures explain how to restart the local service', async () => {
  const h = harness(); h.el('analyze').click(); h.requests[0].reject(vm.runInContext('new TypeError("Failed to fetch")', h.context)); await flush();
  assert.match(h.el('status').textContent, /启动.cmd.*重试/); assert.equal(h.el('result').hidden, true);
});
await test('incomplete success responses are rejected rather than showing a guess', async () => {
  const h = harness(); h.el('analyze').click(); const data = success(); data.analysis.best = null;
  h.requests[0].resolve(data); await flush(); assert.equal(h.el('result').hidden, true); assert.match(h.el('status').textContent, /没有得到可用的出牌建议/);
});
await test('responses without a hand for automatic readback are rejected', async () => {
  const h = harness(); h.el('analyze').click(); const data = success(); delete data.tiles; delete data.hand;
  h.requests[0].resolve(data); await flush(); assert.equal(h.el('result').hidden, true); assert.match(h.el('status').textContent, /可核对的手牌/);
});
await test('the timeout unlocks retry and obsolete responses cannot overwrite a new result', async () => {
  const h = harness(); h.el('analyze').click(); h.tick(45000); assert.equal(h.requests[0].options.signal.aborted, true); assert.equal(h.el('analyze').disabled, false);
  assert.match(h.el('status').textContent, /超时.*重试/); h.el('analyze').click(); const newer = success(); newer.analysis.best = { tile: '东', index: 27, shanten: 1, ukeire: 10, kinds: 3 };
  h.requests[1].resolve(newer); await flush(); h.requests[0].resolve(success()); await flush(); assert.equal(h.el('recommendation').textContent, '建议打 东');
});
await test('leaving the page invalidates a pending response', async () => {
  const h = harness(); h.el('analyze').click(); h.pagehide(); h.requests[0].resolve(success()); await flush();
  assert.equal(h.el('result').hidden, true); assert.equal(h.el('analyze').disabled, false); assert.equal(h.requests[0].options.signal.aborted, true);
});
await test('untrusted window titles and hand strings are rendered as plain text', async () => {
  const h = harness(); h.el('analyze').click(); const data = success(); delete data.tiles; data.hand = '<img src=x onerror=alert(1)>'; data.window.title = '<script>alert(1)</script>';
  h.requests[0].resolve(data); await flush(); assert.equal(h.el('hand').textContent, data.hand); assert.match(h.el('status').textContent, /<script>/);
  assert.doesNotMatch(source, /\.innerHTML\s*=/);
});
console.log('\n通过 ' + passed + ' / ' + passed + ' 一键界面回归测试');
