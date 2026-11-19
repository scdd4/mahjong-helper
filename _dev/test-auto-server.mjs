/** One-click API contracts and isolation. All captures are injected fakes. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createApp } from '../server.mjs';

const HAND = '123456789m2345p4z';
const TILES = [0,1,2,3,4,5,6,7,8,10,11,12,13,30];
function recognized(extra = {}) {
  return {ok:true, hand:HAND, tiles:TILES, confidence:0.94,
    window:{title:'Offline Mahjong fixture',width:1920,height:1080}, ...extra};
}
async function usingServer(options, fn) {
  const server = createApp({...options, autoImage:options.autoImage ?? ''});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = async (body = {}, headers = {}) => {
    const r = await fetch(base + '/api/analyze-window', {method:'POST',headers:{'Content-Type':'application/json',...headers},body:JSON.stringify(body)});
    return {status:r.status, data:await r.json()};
  };
  try { await fn({base,post}); }
  finally { server.closeAllConnections?.(); await new Promise(resolve => server.close(resolve)); }
}

test('default home is the single-button automatic page and never captures on load', async () => {
  let calls = 0;
  await usingServer({autoWorker:{executable:'fake',run:async () => {calls++;return recognized();}}}, async ({base}) => {
    const page = await (await fetch(base)).text();
    assert.equal((page.match(/<button\b/g)||[]).length,1);
    assert.doesNotMatch(page,/<(?:input|select|textarea)\b/i);
    assert.match(page,/截图分析/);
    const health = await (await fetch(base + '/api/health')).json();
    assert.equal(health.autoAvailable,true);
    assert.equal(calls,0);
  });
});

test('missing Python produces a useful error without a capture', async () => {
  await usingServer({}, async ({post}) => {
    const r = await post(); assert.equal(r.status,503); assert.equal(r.data.code,'python_missing');
  });
});

test('one request runs capture, recognition, and solver with no confirmation step', async () => {
  const requests = [];
  await usingServer({autoWorker:{executable:'fake',run:async p => {requests.push(p);return recognized();}}}, async ({post}) => {
    const r = await post(); assert.equal(r.status,200);
    assert.deepEqual(requests,[{mode:'analyze-window'}]);
    assert.equal(r.data.analysis.best.tile,'北'); assert.equal(r.data.analysis.best.ukeire,6);
    assert.equal(r.data.analysis.total,14); assert.deepEqual(r.data.tiles,TILES);
    assert.equal(r.data.analysis.waits.length,2); assert.equal(r.data.hand,HAND);
  });
});

test('13-tile and completed hands do not invent a discard', async () => {
  const hands = [recognized({hand:'123456789m123p1z',tiles:[0,1,2,3,4,5,6,7,8,9,10,11,27]}),
    recognized({hand:'112233m445566p77s',tiles:[0,0,1,1,2,2,12,12,13,13,14,14,24,24]})];
  await usingServer({autoWorker:{executable:'fake',run:async () => hands.shift()}}, async ({post}) => {
    const first = (await post()).data; assert.equal(first.analysis.total,13);
    assert.equal(first.analysis.best,null); assert.equal(first.analysis.tenpai,true);
    const second = (await post()).data; assert.equal(second.analysis.win,true); assert.equal(second.analysis.best,null);
  });
});

test('inconsistent or impossible recognized tiles are never returned as advice', async () => {
  const invalid = [recognized({tiles:TILES.slice(1)}), recognized({hand:'11111m123p123s123z'}),
    recognized({hand:'123m456p789s11z',tiles:[0,1,2,12,13,14,24,25,26,27,27]}),
    recognized({tiles:TILES.slice(0,-1).concat(99)})];
  await usingServer({autoWorker:{executable:'fake',run:async () => invalid.shift()}}, async ({post}) => {
    for (let i=0;i<4;i++) {const r=await post();assert.equal(r.status,422);assert.equal(r.data.ok,false);assert.equal(r.data.analysis,undefined);}
  });
});

test('recognition errors preserve the failure reason and do not contain stale advice', async () => {
  await usingServer({autoWorker:{executable:'fake',run:async () => ({ok:false,code:'window_minimized',error:'请恢复雀魂窗口'})}}, async ({post}) => {
    const r=await post();assert.equal(r.status,422);assert.equal(r.data.code,'window_minimized');assert.equal(r.data.analysis,undefined);
  });
});

test('a running scan blocks a second scan and releases the lock after completion', async () => {
  let finish, started;
  const entered = new Promise(resolve => {started=resolve;});
  await usingServer({autoWorker:{executable:'fake',run:async () => {started();return new Promise(resolve => {finish=resolve;});}}}, async ({post,base}) => {
    const first=post(); await entered;
    assert.equal((await post()).status,429);
    assert.equal((await (await fetch(base+'/api/health')).json()).autoBusy,true);
    finish(recognized());assert.equal((await first).status,200);
    assert.equal((await (await fetch(base+'/api/health')).json()).autoBusy,false);
  });
});

test('other websites and client-selected images or window handles cannot trigger capture', async () => {
  let calls=0;
  await usingServer({autoWorker:{executable:'fake',run:async () => {calls++;return recognized();}}}, async ({post}) => {
    assert.equal((await post({}, {Origin:'https://example.invalid'})).status,403);
    assert.equal((await post({image:'C:/private.png'})).status,400);
    assert.equal((await post({hwnd:123})).status,400);
    assert.equal(calls,0);
  });
});

test('worker errors release the scan lock so the next request can recover', async () => {
  let calls=0;
  await usingServer({autoWorker:{executable:'fake',run:async () => {if (++calls===1) throw new Error('worker failed');return recognized();}}}, async ({post}) => {
    assert.equal((await post()).status,500);assert.equal((await post()).status,200);
  });
});

test('only a trusted test environment can supply an offline fixture', async () => {
  let request;
  await usingServer({autoImage:'fixture.png',autoWorker:{executable:'fake',run:async p => {request=p;return recognized();}}}, async ({post}) => {
    assert.equal((await post()).status,200);assert.deepEqual(request,{mode:'analyze-window',image:'fixture.png'});
  });
});

test('recognized red five and physical tile position survive the analysis response', async () => {
  const codes=['1m','2m','3m','4m','5m','6m','7m','8m','9m','1p','2p','3p','1z','0s'];
  const tiles=[0,1,2,3,4,5,6,7,8,9,10,11,27,22];
  await usingServer({autoWorker:{executable:'fake',run:async () => recognized({hand:codes.join(''),tiles,recognized_tiles:codes.map(name=>({name}))})}},async ({post}) => {
    const r=await post();assert.equal(r.status,200);assert.deepEqual(r.data.tileCodes,codes);
    assert.equal(r.data.analysis.best.index,22);assert.equal(r.data.analysis.best.red,true);
    assert.equal(r.data.analysis.best.position,14);
  });
});
