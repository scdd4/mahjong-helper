/** 本地页面、求解器、可选模型解释与截图识别服务。Node.js 18+。 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { exec, spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const DIR = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(DIR, '雀魂友人场助手.html');
const autoHtmlPath = path.join(DIR, '一键分析.html');
const coreMatch = fs.readFileSync(htmlPath, 'utf8').match(/<script id="mj-core">([\s\S]*?)<\/script>/);
if (!coreMatch) throw new Error('HTML 中未找到 mj-core 求解器');
const MJ = new Function(coreMatch[1] + '\nreturn MJ;')();
const SYSTEM_PROMPT = '你是日麻（立直麻雀）教练。输入是牌效率求解器算好的结论。'
  + '请用一句不超过 25 个字的中文，告诉刚学牌的人这一手为什么这样打。'
  + '输入里括号中的形状（两面/嵌张/边张/单骑/双碰）是权威事实，必须原样使用，不要自己推断。'
  + '不要复述数字，不要编造输入里没有的信息，不要用「根据分析」这类废话开头。';

export function detectPython() {
  const bundled = path.join(os.homedir(), '.dsh', 'dsh-runtimes', 'dsh-primary-runtime', 'dependencies', 'python', 'python.exe');
  for (const executable of new Set([process.env.PYTHON, 'python', 'python3', 'py', bundled].filter(Boolean))) {
    try {
      const r = spawnSync(executable, ['-c', 'import PIL, numpy; print("ok")'], {
        encoding: 'utf8', timeout: 10000, windowsHide: true,
      });
      if (r.status === 0 && /\bok\b/.test(r.stdout || '')) return executable;
    } catch { /* 尝试下一个解释器 */ }
  }
  return null;
}

/** 协议逐行串行；响应只属于发出对应请求的进程和调用。 */
export class PythonWorker {
  constructor(executable, { script = path.join(DIR, 'capture.py'), timeout = 20000, spawnImpl = spawn } = {}) {
    Object.assign(this, { executable, script, timeout, spawnImpl, proc: null, pending: null,
      tail: Promise.resolve(), queued: 0, closed: false });
  }
  run(payload) {
    if (this.closed) return Promise.reject(new Error('识别服务已关闭'));
    if (!this.executable) return Promise.reject(new Error('找不到带 Pillow/numpy 的 Python；请设置 PYTHON'));
    if (this.queued >= 8) return Promise.reject(new Error('识别请求过多，请稍后重试'));
    this.queued++;
    const task = this.tail.then(() => this.execute(payload)).finally(() => { this.queued--; });
    this.tail = task.catch(() => {});
    return task;
  }
  start() {
    const proc = this.spawnImpl(this.executable, ['-B', this.script, '--serve'], { cwd: DIR, windowsHide: true });
    this.proc = proc;
    let buffer = '';
    proc.stdout.setEncoding('utf8');
    proc.stderr.setEncoding('utf8');
    proc.stderr.on('data', d => { if (process.env.DEBUG_PY) process.stderr.write(d); });
    const fail = error => {
      if (this.proc !== proc) return;
      this.proc = null;
      const pending = this.pending;
      this.pending = null;
      if (pending) { clearTimeout(pending.timer); pending.reject(error); }
      try { proc.kill(); } catch { /* 进程已结束 */ }
    };
    proc.on('error', e => fail(new Error('无法启动识别进程：' + e.message)));
    proc.stdin.on('error', e => fail(new Error('识别进程输入失败：' + e.message)));
    proc.on('exit', () => fail(new Error('识别进程退出，请重试')));
    proc.stdout.on('data', d => {
      if (this.proc !== proc) return;
      buffer += d;
      if (buffer.length > 32 * 1024 * 1024) return fail(new Error('识别输出过大'));
      let index;
      while ((index = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);
        if (!line || !this.pending) continue;
        const pending = this.pending;
        try {
          const result = JSON.parse(line);
          if (!result || typeof result !== 'object' || Array.isArray(result)) throw new Error();
          this.pending = null;
          clearTimeout(pending.timer);
          pending.resolve(result);
        } catch { fail(new Error('识别输出异常，请重试')); }
      }
    });
    return proc;
  }
  execute(payload) {
    if (this.closed) return Promise.reject(new Error('识别服务已关闭'));
    return new Promise((resolve, reject) => {
      let proc;
      try { proc = this.proc || this.start(); }
      catch (e) { reject(new Error('无法启动识别进程：' + e.message)); return; }
      const timer = setTimeout(() => {
        if (this.proc !== proc) return;
        this.proc = null;
        this.pending = null;
        try { proc.kill(); } catch { /* ignore */ }
        reject(new Error('识别超时，请重试'));
      }, this.timeout);
      this.pending = { resolve, reject, timer };
      try { proc.stdin.write(JSON.stringify(payload) + '\n'); }
      catch (e) {
        clearTimeout(timer); this.pending = null; this.proc = null;
        try { proc.kill(); } catch { /* ignore */ }
        reject(e);
      }
    });
  }
  close() {
    this.closed = true;
    const proc = this.proc;
    this.proc = null;
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.reject(new Error('识别服务已关闭'));
      this.pending = null;
    }
    if (proc) { try { proc.kill(); } catch { /* ignore */ } }
  }
}

async function askModel(payload, model) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 20000);
  const t0 = performance.now();
  try {
    const headers = { 'Content-Type': 'application/json' };
    if (model.key) headers.Authorization = 'Bearer ' + model.key;
    const res = await fetch(model.url, {
      method: 'POST', headers, signal: ctrl.signal,
      body: JSON.stringify({ model: model.name, messages: [
        { role: 'system', content: SYSTEM_PROMPT }, { role: 'user', content: JSON.stringify(payload) },
      ], max_tokens: 48, stream: false, reasoning_effort: 'none' }),
    });
    if (!res.ok) throw new Error('模型服务返回 HTTP ' + res.status);
    const j = await res.json();
    const content = j.choices?.[0]?.message?.content;
    if (typeof content !== 'string' || !content.trim()) throw new Error('模型没有返回解释文字');
    return { text: content.trim(), ms: Math.round(performance.now() - t0),
      tokens: j.usage?.completion_tokens ?? 0, cached: j.usage?.prompt_tokens_details?.cached_tokens ?? 0,
      timings: j.timings || null };
  } finally { clearTimeout(timer); }
}

function json(res, code, obj) {
  if (res.destroyed || res.writableEnded) return;
  const buf = Buffer.from(JSON.stringify(obj), 'utf8');
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': buf.length,
    'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
  res.end(buf);
}
function readBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0, failed = false;
    const chunks = [];
    req.on('data', c => {
      if (failed) return;
      size += c.length;
      if (size > 256 * 1024) {
        failed = true;
        reject(Object.assign(new Error('请求内容超过 256 KiB'), { status: 413 }));
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => {
      if (failed) return;
      try {
        const result = JSON.parse(Buffer.concat(chunks).toString('utf8').replace(/^\uFEFF/, '') || '{}');
        if (!result || typeof result !== 'object' || Array.isArray(result)) throw new Error();
        resolve(result);
      } catch { reject(Object.assign(new Error('请求必须是合法的 JSON 对象'), { status: 400 })); }
    });
    req.on('aborted', () => reject(Object.assign(new Error('请求中断'), { status: 400 })));
    req.on('error', reject);
  });
}

export function createApp({ worker, autoWorker, model = {}, capImage = process.env.CAP_IMAGE,
  autoImage = process.env.MJ_AUTO_IMAGE,
  calibPath = process.env.MJ_CALIB || path.join(DIR, 'calib.json') } = {}) {
  let llmBusy = false, autoBusy = false;
  const server = http.createServer(async (req, res) => {
    try {
      const port = server.address()?.port;
      const hosts = new Set([`127.0.0.1:${port}`, `localhost:${port}`, `[::1]:${port}`]);
      if (port === 80) for (const host of ['127.0.0.1', 'localhost', '[::1]']) hosts.add(host);
      if (!hosts.has((req.headers.host || '').toLowerCase())) return json(res, 403, { error: '只接受本机地址访问' });
      const origins = new Set([...hosts].map(host => 'http://' + host));
      if (req.headers.origin && !origins.has(req.headers.origin.toLowerCase())) return json(res, 403, { error: '不接受其他网站发起的请求' });
      if (req.headers['sec-fetch-site'] === 'cross-site') return json(res, 403, { error: '不接受跨站请求' });
      if (!req.url?.startsWith('/') || req.url.startsWith('//')) return json(res, 400, { error: '请求地址无效' });
      const url = new URL(req.url, 'http://127.0.0.1');
      if (req.method === 'POST' && !/^application\/json(?:\s*;|$)/i.test(req.headers['content-type'] || '')) return json(res, 415, { error: '请使用 application/json' });
      if (url.pathname === '/' || url.pathname === '/index.html' || url.pathname === '/legacy') {
        if (req.method !== 'GET' && req.method !== 'HEAD') return json(res, 405, { error: '仅支持 GET/HEAD' });
        const buf = fs.readFileSync(url.pathname === '/legacy' ? htmlPath : autoHtmlPath);
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Content-Length': buf.length,
          'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer' });
        res.end(req.method === 'HEAD' ? undefined : buf);
        return;
      }
      if (url.pathname === '/api/health') {
        if (req.method !== 'GET') return json(res, 405, { error: '仅支持 GET' });
        return json(res, 200, { ok: true, model: model.name || null, modelConfigured: !!model.url,
          captureAvailable: !!worker?.executable, autoAvailable: !!autoWorker?.executable,
          autoBusy, busy: llmBusy });
      }
      if (!['/api/solve', '/api/explain', '/api/capture', '/api/learn', '/api/analyze-window'].includes(url.pathname)) return json(res, 404, { error: 'not found' });
      if (req.method !== 'POST') return json(res, 405, { error: '仅支持 POST' });
      if (url.pathname === '/api/analyze-window') {
        if (!autoWorker?.executable) return json(res, 503, { ok: false, code: 'python_missing', error: '自动识别需要 Python、Pillow、numpy 和 OpenCV，请重新运行启动程序' });
        if (autoBusy) return json(res, 429, { ok: false, code: 'busy', error: '上一张截图仍在分析，请稍候再试' });
        autoBusy = true;
        try {
          const input = await readBody(req);
          if (Object.keys(input).length) return json(res, 400, { ok: false, code: 'invalid_request', error: '截图分析无需填写参数' });
          const started = performance.now();
          const request = { mode: 'analyze-window' };
          if (autoImage) request.image = autoImage;
          const out = await autoWorker.run(request);
          if (!out?.ok) return json(res, 422, { ok: false, code: out?.code || 'recognition_failed', error: out?.error || '未识别到完整手牌，请等牌面稳定后重试' });
          try {
            if (typeof out.hand !== 'string' || !Array.isArray(out.tiles)) throw new Error();
            const counts = MJ.parseTiles(out.hand);
            const expected = MJ.tilesOf(counts);
            const actual = out.tiles.slice().sort((a, b) => a - b);
            if (expected.length !== actual.length || expected.some((tile, i) => tile !== actual[i])) throw new Error();
            if (out.tiles.some(tile => !Number.isInteger(tile) || tile < 0 || tile > 33)) throw new Error();
            const result = MJ.analyze(counts, {});
            const best = result.discards[0];
            const details = Array.isArray(out.recognized_tiles) ? out.recognized_tiles : [];
            const tileCodes = details.length === out.tiles.length && details.every((tile, i) => {
              if (typeof tile?.name !== 'string') return false;
              try { const c = MJ.parseTiles(tile.name); return MJ.sum(c) === 1 && c[out.tiles[i]] === 1; }
              catch { return false; }
            }) ? details.map(tile => tile.name) : [];
            const matches = best ? out.tiles.map((tile, i) => tile === best.tile ? i : -1).filter(i => i >= 0) : [];
            const position = matches.find(i => !tileCodes[i]?.startsWith('0')) ?? matches[0];
            const waits = (result.waits?.length ? result.waits : best?.waits || []).map(w => ({ tile: MJ.NAMES[w.tile], index: w.tile, left: w.remaining }));
            return json(res, 200, {
              ok: true, hand: out.hand, tiles: out.tiles, tileCodes, confidence: out.confidence ?? null,
              window: out.window || {}, ms: Math.round(performance.now() - started),
              analysis: { total: result.total, shanten: result.shanten, win: result.win,
                tenpai: result.tenpai || !!best?.tenpai,
                best: best ? { tile: MJ.NAMES[best.tile], index: best.tile, shanten: best.shanten,
                  ukeire: best.ukeireTotal, kinds: best.ukeireKinds, tenpai: !!best.tenpai,
                  position: position + 1, red: !!tileCodes[position]?.startsWith('0') } : null,
                waits },
            });
          } catch {
            return json(res, 422, { ok: false, code: 'invalid_hand', error: '截图中的手牌未能完整可靠识别，请等手牌静止后重试' });
          }
        } finally { autoBusy = false; }
      }
      if (url.pathname === '/api/explain') {
        if (!model.url) return json(res, 503, { error: '模型解释未配置；请设置 SPARK_URL 和 SPARK_MODEL（有鉴权时再设置 SPARK_KEY）' });
        if (llmBusy) return json(res, 429, { error: '模型单并发，上一个请求还没完' });
        llmBusy = true;
        try { return json(res, 200, await askModel(await readBody(req), model)); }
        finally { llmBusy = false; }
      }
      const payload = await readBody(req);
      if (url.pathname === '/api/solve') {
        if (typeof payload.hand !== 'string') return json(res, 400, { error: 'hand 必须是手牌字符串' });
        const counts = MJ.parseTiles(payload.hand);
        const total = MJ.sum(counts);
        if (total !== 13 && total !== 14) return json(res, 400, { error: '手牌必须 13 或 14 张' });
        const t0 = performance.now();
        const r = MJ.analyze(counts, {});
        const best = r.discards[0];
        const waits = (r.waits?.length ? r.waits : best?.waits || []).map(w => ({ tile: MJ.NAMES[w.tile], left: w.remaining }));
        return json(res, 200, { ms: +(performance.now() - t0).toFixed(3), shanten: r.shanten, win: r.win,
          tenpai: r.tenpai || !!best?.tenpai,
          best: best ? { tile: MJ.NAMES[best.tile], ukeire: best.ukeireTotal, tenpai: !!best.tenpai } : null, waits });
      }
      if (!worker?.executable) return json(res, 503, { ok: false, error: '找不到带 Pillow/numpy 的 Python；请设置 PYTHON' });
      if (url.pathname === '/api/learn') {
        const out = await worker.run({ mode: 'learn', assign: payload.assign, n: payload.n, scanId: payload.scanId });
        return json(res, out.ok ? 200 : 400, out);
      }
      const mode = payload.mode || 'probe';
      if (!['probe', 'scan'].includes(mode)) return json(res, 400, { ok: false, error: 'capture 仅支持 probe/scan' });
      const request = { mode, box: payload.box, n: payload.n ?? 14, doraBox: payload.doraBox, discardBoxes: payload.discardBoxes };
      if (capImage) request.image = capImage;
      const out = await worker.run(request);
      if (out.ok && mode === 'scan' && (payload.box || payload.doraBox || payload.discardBoxes)) {
        let prev = {};
        try { prev = JSON.parse(fs.readFileSync(calibPath, 'utf8')); } catch { /* 首次标定 */ }
        const next = { box: out.box || prev.box, n: out.n || prev.n || 14,
          doraBox: out.doraBox ?? prev.doraBox, discardBoxes: out.discardBoxes ?? prev.discardBoxes };
        const temporary = calibPath + '.tmp';
        fs.writeFileSync(temporary, JSON.stringify(next));
        fs.renameSync(temporary, calibPath);
      }
      return json(res, out.ok ? 200 : 400, out);
    } catch (e) { json(res, e.status || (req.url?.startsWith('/api/solve') ? 400 : 500), { ok: false, error: e.message || String(e) }); }
  });
  server.requestTimeout = 30000;
  server.headersTimeout = 15000;
  return server;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 7788);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT 必须是 1 到 65535 的整数');
  const model = { url: process.env.SPARK_URL || '', key: process.env.SPARK_KEY || '',
    name: process.env.SPARK_MODEL || 'qwen3.8-flash-next-iq3_s' };
  if (model.url && !['http:', 'https:'].includes(new URL(model.url).protocol)) throw new Error('SPARK_URL 必须使用 http/https');
  const python = detectPython();
  const worker = new PythonWorker(python);
  const autoWorker = new PythonWorker(python, { script: path.join(DIR, 'auto_worker.py'), timeout: 30000 });
  const server = createApp({ worker, autoWorker, model });
  server.listen(port, '127.0.0.1', () => {
    console.log(`\n  雀魂友人场助手已启动\n  打开: http://127.0.0.1:${port}\n  求解器: 本地 JS`);
    console.log('  模型解释: ' + (model.url ? '已配置 ' + model.name : '未配置（不影响手牌分析）'));
    console.log('  一键截图分析: ' + (autoWorker.executable ? '自动读取雀魂窗口' : '未找到 Python + Pillow/numpy，请设置 PYTHON'));
    console.log('  Ctrl+C 结束\n');
    if (!process.env.NO_OPEN && process.platform === 'win32') exec(`start "" "http://127.0.0.1:${port}/"`, () => {});
  });
  const closeWorkers = () => { worker.close(); autoWorker.close(); };
  process.on('exit', closeWorkers);
  process.on('SIGINT', () => { closeWorkers(); server.close(); process.exit(0); });
  process.on('SIGTERM', () => { closeWorkers(); server.close(); process.exit(0); });
  server.on('error', e => {
    closeWorkers();
    console.error(e.code === 'EADDRINUSE' ? `端口 ${port} 已被占用；请检查是否已经启动，或设置 PORT 换个端口。` : e.message);
    process.exitCode = 1;
  });
}
