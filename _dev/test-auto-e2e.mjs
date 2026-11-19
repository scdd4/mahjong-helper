/** Full offline Python-process -> recognition -> solver integration; no network or desktop. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { detectPython, PythonWorker } from '../server.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const python=detectPython();
assert.ok(python,'Python unavailable');
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'mahjong-auto-offline-'));
const worker=new PythonWorker(python,{script:path.join(root,'auto_worker.py'),timeout:30000});
try {
  const build=spawnSync(python,['-B','-c',[
    'import importlib.util,sys',
    'from pathlib import Path',
    'root=Path(sys.argv[1]); target=Path(sys.argv[2])',
    'spec=importlib.util.spec_from_file_location("fixture_builder",root/"_dev"/"test-auto-recognition.py")',
    'module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)',
    '[module.fixture(n).save(target/(str(n)+".png")) for n in (13,14)]',
  ].join('\n'),root,temporary],{encoding:'utf8',windowsHide:true,timeout:30000});
  assert.equal(build.status,0,build.stderr||build.error?.message);
  const requests=[13,14].map(n=>({mode:'analyze-window',image:path.join(temporary,n+'.png')}));
  const results=await Promise.all(requests.map(p=>worker.run(p)));
  const html=fs.readFileSync(path.join(root,'雀魂友人场助手.html'),'utf8');
  const MJ=new Function(html.match(/<script id="mj-core">([\s\S]*?)<\/script>/)[1]+'\nreturn MJ;')();
  results.forEach((result,i)=>{
    assert.equal(result.ok,true,result.error);
    assert.equal(result.window.method,'fixture');
    assert.equal(result.tiles.length,13+i);
    assert.equal(result.red[0],true);
    const analysis=MJ.analyze(MJ.parseTiles(result.hand),{});
    assert.equal(analysis.total,13+i);
    if (i===1) assert.ok(analysis.win||analysis.discards.length>0);
  });
  const missing=await worker.run({mode:'analyze-window',image:path.join(temporary,'missing.png')});
  assert.equal(missing.ok,false); assert.match(missing.error,/未截取其他画面/);
  const recovered=await worker.run(requests[1]);
  assert.equal(recovered.ok,true); assert.equal(recovered.hand,results[1].hand);
  console.log('离线完整链路通过：真实常驻进程 → 自动识牌13/14张 → 求解器；图片缺失不读屏且可恢复。');
} finally {
  worker.close();
  const tempPrefix=path.resolve(os.tmpdir())+path.sep;
  if (!path.resolve(temporary).startsWith(tempPrefix)) throw new Error('Temporary cleanup path outside temp directory');
  fs.rmSync(temporary,{recursive:true,force:true});
}
