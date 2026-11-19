/** UI state regressions. Runs against the actual inline script with a minimal DOM.
 * No browser, desktop screenshot, model service, or third-party dependency is used.
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const DIR = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(DIR, '..', '雀魂友人场助手.html'), 'utf8');
const script = (id) => html.match(new RegExp('<script id="' + id + '">([\\s\\S]*?)<\\/script>'))[1];

function harness(search = '') {
  const elements = new Map(), timers = new Map(), requests = [];
  let now = 0, timerId = 0;
  function el(id) {
    if (elements.has(id)) return elements.get(id);
    const classes = new Set();
    const node = {
      value: ['seat', 'round'].includes(id) ? '27' : id === 'capN' ? '14' : '',
      checked: false, hidden: false, disabled: false, textContent: '', innerHTML: '', src: '', style: {}, dataset: {}, listeners: {},
      classList: { add: (c) => classes.add(c), remove: (c) => classes.delete(c), toggle(c, on) { const b = on ?? !classes.has(c); b ? classes.add(c) : classes.delete(c); return b; } },
      addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
      fire(type, event = {}) { for (const fn of this.listeners[type] || []) fn.call(this, event); },
      querySelectorAll: () => [], querySelector: () => ({ textContent: '' }), focus() {},
      getBoundingClientRect: () => ({ left: 0, top: 0, right: 1280, bottom: 720, width: 1280, height: 720 }),
    };
    elements.set(id, node); return node;
  }
  const context = vm.createContext({
    document: { getElementById: el, querySelectorAll: () => [], body: el('body') },
    location: { search }, URLSearchParams, AbortController, performance,
    setTimeout(fn, delay) { const id = ++timerId; timers.set(id, { fn, at: now + delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
    fetch(url, options) { return new Promise((resolve, reject) => requests.push({ url, options, resolve: (json, ok = true) => resolve({ ok, json: async () => json }), reject })); },
  });
  vm.runInContext(script('mj-core'), context);
  const names = 'state,cap,parseNow,refreshOpts,doraInfo,fmtHan,renderResult,hideLLM,maybeExplain,capScan,capStart,capRender,capSetCluster,capSetCell,capHandText';
  vm.runInContext(script('mj-ui').replace(/\}\)\(\);\s*$/, 'globalThis.uiTest = {' + names + '};\n})();'), context);
  return {
    ...context.uiTest, MJ: context.MJ, el, requests,
    tick(ms) {
      const end = now + ms;
      for (;;) {
        const next = [...timers.entries()].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!next) break;
        now = next[1].at; timers.delete(next[0]); next[1].fn();
      }
      now = end;
    },
  };
}
async function flush() { for (let i = 0; i < 12; i++) await Promise.resolve(); }
function captured(h, hand = '123456789m2345p4z') {
  const tiles = h.MJ.tilesOf(h.MJ.parseTiles(hand));
  h.cap.box = [0, 0, 1000, 100]; h.cap.mode = 'hand'; h.cap.busy = false;
  const cells = tiles.map((label, i) => ({ i, label, rect: [i * 40, 0, 40, 60] }));
  const clusters = [...new Set(tiles)].map((label, id) => ({ id, label, cells: cells.filter((c) => c.label === label).map((c) => c.i), crop: 'data:image/png;base64,test' }));
  h.cap.data = { ok: true, w: 1280, h: 720, cells, clusters, image: 'fresh-scan-image', scanId: 'scan-test', dora: [], discards: {} };
  h.cap.labels = Object.fromEntries(cells.map((c) => [c.i, c.label])); h.cap.dora = [];
  return h.cap.data;
}

let count = 0;
async function test(name, fn) { await fn(); count++; console.log('OK ' + name); }

await test('invalid hand immediately hides the previous recommendation', () => {
  const h = harness('?demo=1'); assert.match(h.el('result').innerHTML, /建议打/);
  h.el('txt').value = '11111m'; h.el('txt').fire('input'); assert.doesNotMatch(h.el('result').innerHTML, /建议打/);
  h.tick(260); assert.match(h.el('result').innerHTML, /超过 4 张/); assert.equal(h.state.payload, null);
});
await test('automatic text parsing creates a useful undo entry', () => {
  const h = harness('?demo=1'); const original = h.MJ.toText(h.state.counts);
  h.el('txt').value = '123456789m123p1z'; h.el('txt').fire('input'); h.tick(260);
  h.el('btnParse').fire('click'); h.el('btnUndo').fire('click'); assert.equal(h.MJ.toText(h.state.counts), original);
});
await test('clear removes an input error and cancels pending parsing', () => {
  const h = harness('?demo=1'); h.el('txt').value = '11111m'; h.parseNow();
  h.el('btnClear').fire('click'); h.tick(500); assert.equal(h.state.inputError, ''); assert.equal(h.MJ.sum(h.state.counts), 0);
});
await test('invalid advanced options block recommendations rather than silently resetting counts', () => {
  const h = harness('?demo=1'); h.el('seen').value = '11111p'; h.el('seen').fire('input'); h.tick(260);
  assert.match(h.el('result').innerHTML, /已见牌.*超过 4 张/); assert.doesNotMatch(h.el('result').innerHTML, /建议打/);
});
await test('hand plus visible tiles cannot exceed four physical copies', () => {
  const h = harness('?demo=1'); h.el('seen').value = '1111m'; h.refreshOpts();
  assert.match(h.el('result').innerHTML, /合计超过 4 张/);
});
await test('indicator tiles are separately deducted from remaining waits exactly once', () => {
  const h = harness('?demo=1'); h.el('doraInd').value = '2p'; h.el('seen').value = '2p'; h.refreshOpts();
  assert.match(h.state.payload.waits.find((w) => w.startsWith('2筒')), /1张\)$/);
});
await test('combined hand, seen, and indicator copies reject physically impossible input', () => {
  const h = harness('?demo=1'); h.el('doraInd').value = '11m'; h.el('seen').value = '11m'; h.refreshOpts();
  assert.match(h.el('result').innerHTML, /宝牌指示牌.*合计超过 4 张/); assert.equal(h.state.payload, null);
});
await test('duplicate indicators multiply dora; red fives are selected by suit', () => {
  const h = harness(); h.state.doraInd = h.MJ.parseTiles('11m');
  assert.equal(h.doraInfo(h.MJ.parseTiles('22m')).dora, 4);
  h.state.doraInd = h.MJ.zero(); h.state.red = [false, true, false];
  assert.equal(h.doraInfo(h.MJ.parseTiles('5m5p5s')).red, 1);
});
await test('round wind reaches the solver options', () => {
  const h = harness('?demo=1'); h.el('round').value = '28'; h.refreshOpts(); assert.equal(h.state.roundWind, 28);
});
await test('multiple yakuman are labelled with their multiplier', () => {
  const h = harness(); assert.equal(h.fmtHan({ yakuman: 3 }), '3 倍役满');
});
await test('disabled URL flags do not enable model calls or load a demo', () => {
  const h = harness('?demo=0&llm=0&mini=0'); h.tick(1000); assert.equal(h.state.llm, false); assert.equal(h.MJ.sum(h.state.counts), 0); assert.equal(h.requests.length, 0);
});
await test('turning the model off cancels the deferred network call', () => {
  const h = harness('?demo=1'); h.el('llm').checked = true; h.el('llm').fire('change');
  h.el('llm').checked = false; h.el('llm').fire('change'); h.tick(1000); assert.equal(h.requests.length, 0);
});
await test('a hidden model response cannot repopulate the explanation', async () => {
  const h = harness('?demo=1&llm=1'); h.tick(550); assert.equal(h.requests.length, 1);
  h.hideLLM(); h.requests[0].resolve({ text: 'stale', ms: 5 }); await flush(); assert.equal(h.el('llmBox').innerHTML, '');
});
await test('recognized tile classes remain available for manual correction', () => {
  const h = harness(); captured(h); h.capRender(); assert.match(h.el('capBody').innerHTML, /已识别的也可点击纠错/);
  const cl = h.cap.data.clusters[0]; h.capSetCluster(cl.id, cl.label); assert.equal(h.cap.assign[cl.id], cl.label);
});
await test('individual corrections can split a mistakenly merged cluster', () => {
  const h = harness(); captured(h); h.cap.data.clusters = [{ id: 0, label: 0, cells: [0, 1], crop: '' }];
  h.capSetCell(0, 30); assert.equal(h.cap.labels[0], 30); assert.notEqual(h.cap.labels[1], 30); assert.equal(h.cap.pending[0], 30);
});
await test('impossible recognized hands cannot be applied', () => {
  const h = harness(); captured(h); h.cap.data.cells.slice(0, 5).forEach((c) => { h.cap.labels[c.i] = 0; });
  assert.equal(h.capHandText(), null); h.capRender(); assert.equal(h.el('capApply').disabled, true);
});
await test('template learning waits for confirmation and uses the matching scanId', () => {
  const h = harness(); captured(h); h.cap.pending = { 0: 0 }; h.capRender(); assert.equal(h.requests.length, 0);
  h.el('capApply').fire('click'); assert.equal(h.requests[0].url, '/api/learn'); assert.equal(JSON.parse(h.requests[0].options.body).scanId, 'scan-test');
});
await test('an empty new capture clears previous-round dora, visible tiles, and red-five flags', () => {
  const h = harness(); captured(h); h.state.doraInd = h.MJ.parseTiles('1m'); h.state.seen = h.MJ.parseTiles('1z'); h.state.red = [true, false, false];
  h.el('capApply').fire('click'); assert.equal(h.MJ.sum(h.state.doraInd), 0); assert.equal(h.MJ.sum(h.state.seen), 0); assert.equal(h.state.red.some(Boolean), false);
});
await test('capture conflicts block the entire apply operation', () => {
  const h = harness('?demo=1'); captured(h); h.cap.data.discards = { right: { tiles: [0, 0, 0, 0] } };
  const previous = h.MJ.toText(h.state.counts); h.el('capApply').fire('click'); assert.match(h.el('capStatus').textContent, /冲突/); assert.equal(h.MJ.toText(h.state.counts), previous);
});
await test('capture validation includes indicator tiles when their result is eligible for import', () => {
  const h = harness('?demo=1'); captured(h); h.cap.dora = [{ label: 0 }, { label: 0 }, { label: 0 }, { label: 0 }];
  h.el('capApply').fire('click'); assert.match(h.el('capStatus').textContent, /宝牌指示牌合计超过 4 张/);
});
await test('discard quality warnings prevent uncertain tiles from entering visible counts', () => {
  const h = harness(); captured(h); h.cap.data.discardNote = 'not reliable'; h.cap.data.discards = { right: { count: 4, tiles: [0, 0, 0, 0] } };
  h.el('capApply').fire('click'); assert.equal(h.MJ.sum(h.state.seen), 0);
});
await test('dora quality warnings keep uncertain indicators out of calculations', () => {
  const h = harness(); captured(h); h.cap.data.doraNote = 'partial'; h.cap.dora = [{ label: 0 }];
  h.el('capApply').fire('click'); assert.equal(h.MJ.sum(h.state.doraInd), 0);
});
await test('exhausted waits are explicitly marked for 13 and 14 tile hands', () => {
  const h = harness('?demo=1'); h.el('seen').value = '222555p444z'; h.refreshOpts(); assert.match(h.el('result').innerHTML, /可见牌范围内已无剩余和牌张/);
  h.el('txt').value = '123456789m2345p'; h.parseNow(); assert.match(h.el('result').innerHTML, /可见牌范围内已无剩余和牌张/);
});
await test('a new scan resets labels and displays the exact scanned image', async () => {
  const h = harness(); const data = captured(h); h.cap.assign = { 0: 30 }; h.cap.pending = { 0: 30 }; h.el('capImg').src = 'old-probe';
  h.capScan(); assert.equal(Object.keys(h.cap.assign).length, 0); assert.equal(h.el('capApply').disabled, true);
  h.requests[0].resolve(data); await flush(); assert.equal(h.el('capImg').src, 'fresh-scan-image'); assert.equal(h.cap.busy, false);
});
await test('failed scans cannot leave an earlier hand applicable', async () => {
  const h = harness(); captured(h); h.capScan(); h.requests[0].resolve({ ok: false, error: 'test failure' }); await flush();
  assert.equal(h.cap.data, null); assert.equal(h.el('capApply').disabled, true); assert.equal(h.capHandText(), null);
});
await test('closing capture ignores an obsolete asynchronous response', async () => {
  const h = harness(); const data = captured(h); h.capScan(); h.el('capClose').fire('click'); h.requests[0].resolve(data); await flush();
  assert.equal(h.el('capModal').hidden, true); assert.equal(h.el('capApply').disabled, true); assert.notEqual(h.el('capImg').src, data.image);
});
console.log('\n通过 ' + count + ' / ' + count + ' UI 状态回归测试');
