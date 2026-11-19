import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HTML = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '雀魂友人场助手.html');
const html = fs.readFileSync(HTML, 'utf8');
const m = html.match(/<script id="mj-core">([\s\S]*?)<\/script>/);
if (!m) { console.error('找不到 mj-core 脚本块'); process.exit(1); }
const MJ = new Function(m[1] + '\nreturn MJ;')();

let pass = 0, fail = 0;
const failures = [];
function ok(cond, label, extra) {
  if (cond) pass++;
  else { fail++; failures.push(label + (extra !== undefined ? '  →  ' + JSON.stringify(extra) : '')); }
}
function eq(a, b, label) { ok(a === b, label, { got: a, want: b }); }

const P = (s) => MJ.parseTiles(s);
const S = (c) => MJ.toText(c);

/* ================= 1. 解析 / 输出 ================= */
eq(S(P('123m456p789s11z')), '123m456p789s11z', '1.1 解析+还原');
eq(S(P('1m1m1m2p')), '111m2p', '1.2 单张写法合并');
eq(MJ.sum(P('19m19p19s1234567z')), 13, '1.3 国士 13 张');
eq(S(P('123万456筒789索东东')), '123m456p789s11z', '1.4 中文花色');
eq(S(P('123456789m111p')), '123456789m111p', '1.5 九莲解析');
let threw = false;
try { P('11111m'); } catch { threw = true; }
ok(threw, '1.6 超过 4 张报错');
threw = false;
try { P('123'); } catch { threw = true; }
ok(threw, '1.7 缺花色报错');
eq(MJ.doraOf(0), 1, '1.8 1万→2万');
eq(MJ.doraOf(8), 0, '1.9 9万→1万');
eq(MJ.doraOf(9 + 8), 9, '1.10 9筒→1筒');
eq(MJ.doraOf(30), 27, '1.11 北→东');
eq(MJ.doraOf(33), 31, '1.12 中→白');
eq(MJ.doraOf(31), 32, '1.13 白→发');

/* ================= 2. 和牌判定 ================= */
ok(MJ.isWin(P('11122233344455m')), '2.1 四刻+对');
ok(MJ.isWin(P('234567m234567p88s')), '2.2 全顺子');
ok(!MJ.isWin(P('234567m234567p89s')), '2.3 差一张不和');
ok(MJ.isWin(P('11223344556677p')), '2.4 七对子');
ok(MJ.isWin(P('19m19p19s1234567z1m')), '2.5 国士无双');
ok(MJ.isWin(P('22334455667788p')), '2.6 七对子/二杯口 都能和');
ok(!MJ.isWin(P('2233445566778p')), '2.6b 13 张不算和牌');
ok(!MJ.isWin(P('19m19p19s1234567z2m')), '2.7 国士缺一种');
ok(!MJ.isWin(P('123456789m1234p')), '2.8 13 张不算和牌');
ok(MJ.isWin(P('123123m456789p11s')), '2.9 一杯口形');

/* ================= 3. 向听数（已知牌型） ================= */
const sh = (s) => MJ.bestShanten(P(s)).value;
eq(sh('123456789m1234p'), 0, '3.1 听牌 0 向听');
eq(sh('123456789m12p45s'), 1, '3.2 13 张：三面子加两搭子 1 向听');
eq(sh('1112223334445m'), 0, '3.3 4 刻+单张 = 听牌');
eq(sh('19m19p19s1234567z'), 0, '3.4 国士十三面 0 向听');
eq(sh('1122334455667m'), 0, '3.5 六对+单 = 七对子听牌');
eq(sh('11223344556677m'), -1, '3.6 七对子和牌 -1 向听');
eq(sh('147m258p369s1234z'), 6, '3.7 全孤立：七对子更近（6 向听）');
eq(sh('19m19p19s123456z2m'), 1, '3.8 国士差一种 1 向听');

/* ================= 4. 听牌张 ================= */
const waits = (s) => MJ.waitsOf(P(s)).map((i) => MJ.NAMES[i]);
eq(waits('123456789m1234p').join(','), '1筒,4筒', '4.1 两面听 1筒/4筒');
eq(waits('123456789m123p1z').join(','), '东', '4.2 单骑听 东');
ok(waits('19m19p19s1234567z').length === 13, '4.3 国士十三面听');

/* ================= 5. 役判定 ================= */
function score(s, winTile) {
  const c = P(s);
  const t = winTile == null ? null : P(winTile).findIndex((x) => x > 0);
  return MJ.bestScore(c, {}, t);
}
function hasYaku(r, name) { return r.yakus.some((y) => y[0] === name); }

let r = score('234567m234567p88s', '4m');
ok(hasYaku(r, '平和') && hasYaku(r, '断幺九'), '5.1 平和+断幺', r.yakus);
eq(r.han, 2, '5.2 平和+断幺 = 2 番');

r = score('11223344556677m');
ok(hasYaku(r, '二杯口') && hasYaku(r, '清一色'), '5.3 二杯口+清一色', r.yakus);
eq(r.han, 10, '5.4 二杯口 3 + 清一色 6 + 平和 1 = 10 番');

r = score('19m19p19s1234567z1m');
eq(r.yakuman, 1, '5.5 国士是役满');
eq(r.han, 13, '5.6 国士 13 番');

r = score('1112345678999m5m');
ok(hasYaku(r, '九莲宝灯'), '5.7 九莲宝灯', r.yakus);

r = score('123m456p789s234p11z', '2p');
eq(r.han, 0, '5.8 无役手牌 = 0 番', r.yakus);

r = score('123456789m111m11z', '5m');
ok(hasYaku(r, '混一色') && hasYaku(r, '一气通贯'), '5.9 混一色+一气', r.yakus);
eq(r.han, 5, '5.10 混一色 3 + 一气 2 = 5 番');

r = score('123m456m789m123p99s', '4m');
ok(hasYaku(r, '平和'), '5.11 两面和牌有平和', r.yakus);
r = score('123m456m789m123p99s', '9s');
ok(!hasYaku(r, '平和'), '5.12 单骑和牌没有平和', r.yakus);

r = score('123m123p123s456m99p', '4m');
ok(hasYaku(r, '三色同顺') && hasYaku(r, '平和'), '5.13 三色同顺+平和', r.yakus);
eq(r.han, 3, '5.14 三色 2 + 平和 1 = 3 番');

r = score('123m456m789m555z99p', '5m');
ok(hasYaku(r, '役牌 白'), '5.15 白刻子算役牌', r.yakus);

r = score('1122m3344p5566s77z');
ok(hasYaku(r, '七对子'), '5.16 七对子', r.yakus);
eq(r.han, 2, '5.17 七对子 2 番');

r = score('234567m234567p88s', '4m');
eq(r.shape, '两面', '5.18 听牌形=两面');
r = score('123m456m789m123p99s', '9s');
eq(r.shape, '单骑', '5.19 听牌形=单骑');

/* ================= 6. 一致性属性测试 ================= */
function rndHand(n, rng) {
  for (;;) {
    const c = MJ.zero();
    let left = n, guard = 0;
    while (left > 0 && guard++ < 5000) {
      const i = Math.floor(rng() * 34);
      if (c[i] < 4) { c[i]++; left--; }
    }
    if (left === 0) return c;
  }
}
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(20261118);

/* 6a: 13 张手牌，0 向听 ⟺ 有和牌张 */
let bad = 0;
for (let k = 0; k < 400; k++) {
  const c = rndHand(13, rng);
  const s = MJ.bestShanten(c).value;
  const w = MJ.waitsOf(c);
  if ((s === 0) !== (w.length > 0)) { bad++; if (bad < 3) failures.push('6a 反例 ' + S(c) + ' shanten=' + s + ' waits=' + w.length); }
}
ok(bad === 0, '6a 13 张：0 向听 ⟺ 有和牌张', { bad });

/* 6b: 未和牌的 14 张手牌向听数 = 打掉一张后的最小向听数 */
bad = 0;
for (let k = 0; k < 250; k++) {
  const c = rndHand(14, rng);
  if (MJ.isWin(c)) continue;
  const direct = MJ.bestShanten(c).value;
  let best = 99;
  for (let d = 0; d < 34; d++) {
    if (!c[d]) continue;
    c[d]--;
    const s = MJ.bestShanten(c).value;
    if (s < best) best = s;
    c[d]++;
  }
  if (direct !== best) { bad++; if (bad < 3) failures.push('6b 反例 ' + S(c) + ' direct=' + direct + ' min=' + best); }
}
ok(bad === 0, '6b 14 张向听 = min(打一张后向听)', { bad });

/* 6c: 进张判定必须恰好降低 1 向听 */
bad = 0;
for (let k = 0; k < 120; k++) {
  const c = rndHand(13, rng);
  const s = MJ.bestShanten(c).value;
  if (s === 0) continue;
  const u = new Set(MJ.ukeireList(c, s, c, MJ.zero()).map((x) => x.tile));
  for (let t = 0; t < 34; t++) {
    if (c[t] >= 4) continue;
    c[t]++;
    const s2 = MJ.bestShanten(c).value;
    c[t]--;
    const listed = u.has(t);
    if (s2 > s || s2 < s - 1) { bad++; failures.push('6c 向听跳变 ' + S(c) + ' +' + MJ.NAMES[t]); break; }
    if (listed !== (s2 === s - 1)) { bad++; failures.push('6c 进张不符 ' + S(c) + ' +' + MJ.NAMES[t] + ' s=' + s + ' s2=' + s2); break; }
  }
}
ok(bad === 0, '6c 进张列表正确', { bad });

/* 6d: analyze 的推荐出牌 —— 向听最优性 */
bad = 0;
for (let k = 0; k < 100; k++) {
  const c = rndHand(14, rng);
  if (MJ.isWin(c)) continue;
  const r2 = MJ.analyze(c, {});
  const top = r2.discards[0];
  let best = 99;
  for (let d = 0; d < 34; d++) {
    if (!c[d]) continue;
    c[d]--;
    best = Math.min(best, MJ.bestShanten(c).value);
    c[d]++;
  }
  if (top.shanten !== best) { bad++; failures.push('6d 推荐非最优 ' + S(c)); }
  if (top.ukeireTotal < 0) { bad++; failures.push('6d 进张负数 ' + S(c)); }
}
ok(bad === 0, '6d 推荐出牌 = 最小向听', { bad });

/* ================= 7. 综合示例 ================= */
try {
  const r2 = MJ.analyze(P('123456789m2345p4z'), {});
  const top = r2.discards[0];
  ok(!!top, '7.0 示例有出牌候选', { discards: r2.discards.length, win: r2.win });
  eq(MJ.NAMES[top.tile], '北', '7.1 示例推荐打孤张 北');
  ok(top.tenpai, '7.2 打 北 后听牌');
  eq(top.waits.map((w) => MJ.NAMES[w.tile]).join(','), '2筒,5筒', '7.3 听 2筒/5筒');
  eq(top.waits.reduce((a, w) => a + w.remaining, 0), 6, '7.3b 理论剩 6 张');

  const r3 = MJ.analyze(P('19m19p19s1234567z'), {});
  ok(r3.tenpai && r3.waits.length === 13, '7.4 13 张国士 = 十三面听');

  const r4 = MJ.analyze(P('11122233344455m'), {});
  ok(r4.win, '7.5 完整手牌判定为和牌');

  const r5 = MJ.analyze(P('123456789m2345p4z'), { seen: P('22p') });
  const w5 = r5.discards[0].waits.find((w) => MJ.NAMES[w.tile] === '2筒');
  eq(w5.remaining, 1, '7.6 已见牌扣减剩余枚数');
} catch (e) {
  fail++; failures.push('第 7 节异常：' + e.message);
}

/* ================= 8. 性能 ================= */
let perfMs = 0;
try {
  const benchMJ = new Function(m[1] + '\nreturn MJ;')();
  const c = benchMJ.parseTiles('1234567m2345p678s');
  eq(MJ.sum(c), 14, '8.0 性能牌例确实为 14 张');
  const coldStart = performance.now();
  const first = benchMJ.analyze(c, {});
  const firstMs = performance.now() - coldStart;
  ok(!first.win && first.discards.length > 0 && first.discards[0].ukeireKinds > 0, '8.0b 性能牌例执行完整出牌与进张分析');
  ok(firstMs < 400, '8.0c 首次完整分析含初始化 < 400ms', { ms: Math.round(firstMs) });
  const t0 = performance.now();
  for (let k = 0; k < 20; k++) benchMJ.analyze(c, {});
  perfMs = (performance.now() - t0) / 20;
  ok(perfMs < 400, '8.1 单次分析 < 400ms', { ms: Math.round(perfMs) });
  console.log(`性能：14 张完整分析首次 ${firstMs.toFixed(1)} ms，重复平均 ${perfMs.toFixed(1)} ms`);
  const spread = benchMJ.parseTiles('147m258p369s12345z');
  const spreadStart = performance.now();
  const spreadResult = benchMJ.analyze(spread, {});
  ok(spreadResult.discards.length > 0 && performance.now() - spreadStart < 400, '8.2 14 张全孤立牌首次分析 < 400ms');
} catch (e) {
  fail++; failures.push('第 8 节异常：' + e.message);
}

/* ================= 9. 界面脚本语法检查 ================= */
{
  const ui = html.match(/<script id="mj-ui">([\s\S]*?)<\/script>/);
  try { new Function(ui[1]); pass++; }
  catch (e) { fail++; failures.push('9.1 UI 脚本语法错误：' + e.message); }
}

/* ================= 10. 审查问题回归 ================= */
function rejects(fn, label) {
  let threw = false;
  try { fn(); } catch { threw = true; }
  ok(threw, label);
}
for (const text of ['123x456m', '123m!456p', 'm123p', '123mm', '1.5m']) {
  rejects(() => P(text), '10.1 拒绝静默改写输入：' + text);
}
eq(S(P('123M，456筒 东 东')), '123m456p11z', '10.2 保留英文大写、逗号和空白输入');
eq(sh('1111222333444z'), 1, '10.3 四东加三组风刻子不能听第五张东');
eq(waits('1111222333444z').length, 0, '10.4 第五张东不是合法听牌张');
eq(sh('1111222233334z'), 3, '10.5 三组字牌暗杠的多余牌不能虚当进张');
eq(sh('11112222333344z'), 2, '10.6 14 张三组字牌暗杠实际 2 向听');
const fakeTenpai = MJ.analyze(P('1111222333444z'));
ok(!fakeTenpai.tenpai && fakeTenpai.ukeire.length > 0, '10.7 假听牌改为显示有效进张');

r = score('111123m222p333s55z', '1m');
ok(hasYaku(r, '三暗刻'), '10.8 荣和 1 万放入顺子时同牌刻子仍是暗刻', r);
eq(r.han, 2, '10.9 荣和落点枚举保留三暗刻的 2 番');
r = score('123m456p789s234p11z', '2p');
eq(r.han, 0, '10.10 无役和型仍是 0 番');
eq(r.shape, '两面', '10.11 无役和型保留准确听牌形状');
eq(r.winTile, P('2p').findIndex((x) => x > 0), '10.12 无役和型保留指定和牌张');
r = score('111m111p111s234s55z', '4s');
ok(hasYaku(r, '三色同刻') && hasYaku(r, '三暗刻'), '10.13 三色同刻与三暗刻可复合', r);
eq(r.han, 4, '10.14 三色同刻 2 番 + 三暗刻 2 番');
r = score('11m99m11p99p11s99s11z', '1z');
ok(hasYaku(r, '七对子') && hasYaku(r, '混老头'), '10.15 七对子与混老头可复合', r);
eq(r.han, 4, '10.16 七对子混老头 4 番');
r = score('11223344556677z', '1z');
eq(r.yakuman, 1, '10.17 全字七对子是字一色役满');
ok(hasYaku(r, '字一色'), '10.18 字一色替代普通役番数', r);
r = score('22233344466688s', '8s');
ok(hasYaku(r, '绿一色'), '10.19 支持绿一色', r);
r = score('111222333z444m44z', '4m');
ok(hasYaku(r, '小四喜'), '10.20 支持小四喜', r);
r = score('11122233344455z', '5z');
ok(hasYaku(r, '大四喜') && hasYaku(r, '字一色') && hasYaku(r, '四暗刻单骑'), '10.21 复合役满判定', r);
eq(r.yakuman, 5, '10.22 大四喜双倍 + 字一色 + 四暗刻单骑双倍 = 五倍役满');
r = score('19m19p19s1234567z1m', '1m');
eq(r.yakuman, 2, '10.23 国士十三面双倍役满');
r = score('11123456789995m', '5m');
eq(r.yakuman, 2, '10.24 纯正九莲双倍役满');
r = score('11123456789995m', '1m');
eq(r.yakuman, 1, '10.25 非九面和牌张只是普通九莲');
ok(!!r.shape, '10.25b 普通九莲保留和牌形状');
r = score('11112222333344m', '1m');
ok(hasYaku(r, '二杯口') && hasYaku(r, '清一色') && hasYaku(r, '平和'), '10.25c 四组相同顺子仍有两组杯口', r);
eq(r.han, 10, '10.25d 四组相同顺子：二杯口 3 + 清一色 6 + 平和 1 = 10 番');
const south = MJ.bestScore(P('123456m789p55s222z'), { roundWind: 28, seatWind: 27 }, 22);
ok(hasYaku(south, '场风 南'), '10.26 场风选择南时南刻子是役牌', south);
eq(south.han, 1, '10.27 南场非南家单独计算 1 番');
const east = MJ.bestScore(P('123456m789p55s222z'), { roundWind: 27, seatWind: 27 }, 22);
eq(east.han, 0, '10.28 东场非南家南刻子不是役牌');

rejects(() => MJ.analyze(P('123m')), '10.29 analyze 拒绝非 13/14 张手牌');
rejects(() => MJ.analyze(P('123456789m2345p4z'), { seen: P('2222p') }), '10.30 手牌与已见牌合计不能超过四张');
rejects(() => MJ.analyze(Array(33).fill(0)), '10.31 拒绝错误长度数组');
const invalid = P('123456789m2345p4z'); invalid[0] = NaN;
rejects(() => MJ.analyze(invalid), '10.32 拒绝 NaN 牌数');
rejects(() => MJ.analyze(P('123456789m2345p4z'), { roundWind: 31 }), '10.33 拒绝非风牌场风');
rejects(() => MJ.analyze(P('123456789m2345p4z'), { seatWind: '27' }), '10.34 拒绝非数字自风');
const dead = MJ.analyze(P('1111222333444z'), { seen: P('5555z') });
ok(dead.ukeire.every((u) => u.remaining > 0 && u.tile !== 31), '10.35 已耗尽的牌不计入有效进张');
const before = P('111123m222p333s5z'), snapshot = S(before);
MJ.analyze(before, {});
eq(S(before), snapshot, '10.36 分析与评分不修改传入的手牌数组');

/* ================= 汇总 ================= */
console.log(`\n通过 ${pass} / ${pass + fail}`);
if (fail) {
  console.log('\n失败项：');
  failures.slice(0, 40).forEach((f) => console.log('  ✗ ' + f));
  process.exit(1);
} else {
  console.log('全部通过 ✅');
}

