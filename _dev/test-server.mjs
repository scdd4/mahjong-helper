/** 本机回归测试；不截屏、不访问模型网络、不使用真实凭据。 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import http from 'node:http';
import { EventEmitter } from 'node:events';
import { PassThrough, Writable } from 'node:stream';
import { PythonWorker, createApp } from '../server.mjs';

function mockSpawn(plans = []) {
  let starts = 0;
  const spawnImpl = () => {
    const plan = plans[starts++] || {};
    const proc = new EventEmitter();
    proc.stdout = new PassThrough();
    proc.stderr = new PassThrough();
    proc.stdin = new Writable({ write(chunk, encoding, callback) {
      const payload = JSON.parse(chunk.toString());
      if (plan.error) queueMicrotask(() => proc.emit('error', new Error('spawn failed')));
      else if (plan.stdinError) queueMicrotask(() => proc.stdin.emit('error', new Error('EPIPE')));
      else if (plan.exit) queueMicrotask(() => proc.emit('exit', 1));
      else if (!plan.silent) setTimeout(() => proc.stdout.write(JSON.stringify({ ok: true, id: payload.id }) + '\n'), plan.delay || 0);
      callback();
    } });
    proc.kill = () => { setTimeout(() => proc.emit('exit', 0), plan.exitDelay || 0); return true; };
    return proc;
  };
  return { spawnImpl, starts: () => starts };
}

test('并发识别请求按顺序收到各自响应', async () => {
  const mock = mockSpawn([{ delay: 5 }]);
  const worker = new PythonWorker('fake', { spawnImpl: mock.spawnImpl, timeout: 1000 });
  try {
    const results = await Promise.all([worker.run({ id: 1 }), worker.run({ id: 2 }), worker.run({ id: 3 })]);
    assert.deepEqual(results.map(r => r.id), [1, 2, 3]);
    assert.equal(mock.starts(), 1);
  } finally { worker.close(); }
});

test('超时后新进程恢复，旧进程的退出不会清除新请求', async () => {
  const mock = mockSpawn([{ silent: true, exitDelay: 20 }, { delay: 30 }]);
  const worker = new PythonWorker('fake', { spawnImpl: mock.spawnImpl, timeout: 50 });
  try {
    await assert.rejects(worker.run({ id: 1 }), /超时/);
    assert.equal((await worker.run({ id: 2 })).id, 2);
    assert.equal(mock.starts(), 2);
  } finally { worker.close(); }
});

for (const fault of ['error', 'stdinError', 'exit']) test(fault + ' 故障可恢复，无未处理事件', async () => {
  const mock = mockSpawn([{ [fault]: true }, {}]);
  const worker = new PythonWorker('fake', { spawnImpl: mock.spawnImpl, timeout: 1000 });
  try {
    await assert.rejects(worker.run({ id: 1 }));
    assert.equal((await worker.run({ id: 2 })).id, 2);
  } finally { worker.close(); }
});

test('关闭识别服务会拒绝尚未完成的调用', async () => {
  const mock = mockSpawn([{ silent: true }]);
  const worker = new PythonWorker('fake', { spawnImpl: mock.spawnImpl });
  const first = worker.run({ id: 1 });
  const second = worker.run({ id: 2 });
  worker.close();
  await Promise.all([assert.rejects(first, /关闭/), assert.rejects(second, /关闭/)]);
});

function request(port, route, { method = 'POST', body = '{}', headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({ hostname: '127.0.0.1', port, path: route, method,
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body), ...headers } }, res => {
      let raw = '';
      res.setEncoding('utf8');
      res.on('data', chunk => { raw += chunk; });
      res.on('error', error => { error.message += ` (response ${route}, ${body.length} bytes)`; reject(error); });
      res.on('end', () => resolve({ status: res.statusCode, json: JSON.parse(raw) }));
    });
    req.on('error', error => { error.message += ` (${method} ${route}, ${body.length} bytes, ${JSON.stringify(headers)})`; reject(error); });
    req.end(body);
  });
}
async function listen(server) {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return server.address().port;
}
async function close(server) {
  server.closeIdleConnections?.();
  await new Promise(resolve => server.close(resolve));
}

test('HTTP 拒绝跨站、Host 绕过、非JSON和超大请求；错误后仍可服务', async () => {
  let captures = 0;
  const server = createApp({ worker: { executable: 'fake', run: async () => { captures++; return { ok: true }; } } });
  const port = await listen(server);
  try {
    for (const headers of [{ Origin: 'https://example.invalid' }, { Origin: 'null' },
      { Host: 'example.invalid:' + port }, { 'Sec-Fetch-Site': 'cross-site' }]) {
      assert.equal((await request(port, '/api/capture', { headers })).status, 403);
    }
    assert.equal(captures, 0);
    assert.equal((await request(port, '/api/capture', { headers: { 'Content-Type': 'text/plain' } })).status, 415);
    assert.equal((await request(port, '/api/solve', { body: '{' })).status, 400);
    assert.equal((await request(port, '/api/solve', { body: 'null' })).status, 400);
    assert.equal((await request(port, '/api/solve', { body: JSON.stringify({ hand: 'x'.repeat(270000) }) })).status, 413);
    assert.equal((await request(port, '/api/solve')).status, 400);
    assert.equal((await request(port, '/api/solve', { body: JSON.stringify({ hand: '112233m445566p77s' }) })).status, 200);
    const health = await request(port, '/api/health', { method: 'GET' });
    assert.equal(health.json.modelConfigured, false);
    assert.equal((await request(port, '/api/explain')).status, 503);
    assert.equal((await request(port, '/api/capture', { headers: { Origin: 'http://127.0.0.1:' + port } })).status, 200);
    assert.equal(captures, 1);
  } finally { await close(server); }
});

test('模型单并发锁覆盖请求体读取；模型返回后解锁', async () => {
  let calls = 0;
  const upstream = http.createServer((req, res) => {
    calls++;
    req.resume();
    req.on('end', () => { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({ choices: [{ message: { content: '保留两面形' } }] })); });
  });
  const upstreamPort = await listen(upstream);
  const server = createApp({ model: { url: 'http://127.0.0.1:' + upstreamPort, name: 'test' } });
  const port = await listen(server);
  try {
    let first;
    const result = new Promise((resolve, reject) => {
      first = http.request({ hostname: '127.0.0.1', port, path: '/api/explain', method: 'POST',
        headers: { 'Content-Type': 'application/json' } }, res => { res.resume(); res.on('end', () => resolve(res.statusCode)); });
      first.on('error', reject);
      first.write('{');
    });
    for (let i = 0; i < 100; i++) {
      const r = await request(port, '/api/health', { method: 'GET' });
      if (r.json.busy) break;
      await new Promise(resolve => setTimeout(resolve, 2));
    }
    assert.equal((await request(port, '/api/explain')).status, 429);
    first.end('}');
    assert.equal(await result, 200);
    assert.equal(calls, 1);
    assert.equal((await request(port, '/api/explain', { body: '{' })).status, 400);
    assert.equal((await request(port, '/api/explain')).status, 200);
    assert.equal(calls, 2);
  } finally { await close(server); await close(upstream); }
});
