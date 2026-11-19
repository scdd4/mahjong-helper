import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { detectPython } from '../server.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const python=detectPython();
if (!python) throw new Error('未找到自动识别所需的 Python');
for (const file of ['test-window-capture.py','test-auto-recognition.py']) {
  const r=spawnSync(python,['-B',path.join(root,'_dev',file)],{cwd:root,encoding:'utf8',windowsHide:true,timeout:60000});
  process.stdout.write(r.stdout||''); process.stderr.write(r.stderr||'');
  if (r.error||r.status!==0) throw new Error(file+' 未通过：'+(r.error?.message||r.status));
}
