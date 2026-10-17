/**
 * 端到端验证：`parsePly` 能否从真实的 **3DGS PLY** 里读全 **59 个参数**。
 *
 * 用法: python scripts/dump_gauss_probe.py  →  node scripts/probe_gauss_ply.mjs
 *
 * 为什么需要：`.splat` 只有 11 维、glTF 点云只有 6 维，都**丢了信息**。
 * 要让查看器显示"和预览图一样的模型"，就必须从 3DGS PLY 读到全集：
 *   位置3 + 尺度3 + 旋转4 + 不透明度1 + SH 48 = **59**
 * 任何一维读错（属性名、偏移、类型、通道主序），外观都会错，
 * 而**没有浏览器时看不出来**。这里与 NumPy 的值逐个比对。
 */
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(import.meta.dirname, '..');
const probePath = path.join(ROOT, 'env', 'tmp', 'gauss_probe.json');
if (!fs.existsSync(probePath)) {
  console.error('缺少 probe，请先运行: python scripts/dump_gauss_probe.py');
  process.exit(2);
}
const probe = JSON.parse(fs.readFileSync(probePath, 'utf8'));

// 从 viewer.js 里切出 parsePly（模块私有，这里单独取出以便在 Node 里跑）
const viewerSrc = fs.readFileSync(path.join(ROOT, 'src', 'gsflow', 'web', 'static', 'viewer.js'), 'utf8');
const start = viewerSrc.indexOf('function parsePly(');
if (start < 0) {
  console.error('viewer.js 里找不到 parsePly');
  process.exit(2);
}
let depth = 0, end = -1;
for (let i = start; i < viewerSrc.length; i++) {
  if (viewerSrc[i] === '{') depth++;
  else if (viewerSrc[i] === '}') { depth--; if (depth === 0) { end = i + 1; break; } }
}
const parsePlySrc = viewerSrc.slice(start, end);
const parsePly = new Function(parsePlySrc + '\nreturn parsePly;')();

const buf = fs.readFileSync(probe.ply_path);
const model = parsePly(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));

let bad = 0;
const chk = (cond, msg) => {
  if (!cond) bad++;
  console.log(`  [${cond ? 'OK  ' : 'FAIL'}] ${msg}`);
};

console.log('=== 3DGS PLY 解析（59 维）===');
chk(!!model.gauss3d, '识别为 3DGS（有 gauss3d）');
const g = model.gauss3d || {};
const n = probe.count;
chk(g.pos && g.pos.length === n * 3, `位置 ${n * 3} 个浮点`);
chk(g.scale && g.scale.length === n * 3, `尺度 ${n * 3} 个浮点`);
chk(g.quat && g.quat.length === n * 4, `旋转 ${n * 4} 个浮点`);
chk(g.opacity && g.opacity.length === n, `不透明度 ${n} 个浮点`);
chk(g.shDC && g.shDC.length === n * 3, `SH DC ${n * 3} 个浮点`);
const expectRest = n * probe.n_rest;
chk(g.shRest && g.shRest.length === expectRest, `SH 高阶 ${expectRest} 个浮点`);
chk(g.shDegree === probe.sh_degree, `SH 阶 = ${g.shDegree}（期望 ${probe.sh_degree}）`);

// 逐值比对（前若干个高斯）—— **分字段报告**，否则只看到一个"最大误差"无法定位
console.log('\n=== 数值比对（JS vs NumPy）===');
const fields = { pos: 0, scale: 0, quat: 0, opacity: 0, sh_dc: 0, sh_rest: 0 };
const bump = (key, arr, want) => {
  for (let k = 0; k < want.length; k++) {
    const e = Math.abs(arr[k] - want[k]);
    if (e > fields[key]) fields[key] = e;
  }
};
for (const c of probe.cases) {
  const i = c.idx;
  bump('pos', g.pos.subarray(i * 3, i * 3 + 3), c.pos);
  bump('scale', g.scale.subarray(i * 3, i * 3 + 3), c.scale);
  bump('quat', g.quat.subarray(i * 4, i * 4 + 4), c.quat);
  bump('opacity', g.opacity.subarray(i, i + 1), [c.opacity]);
  bump('sh_dc', g.shDC.subarray(i * 3, i * 3 + 3), c.sh_dc);
  if (probe.n_rest > 0 && g.shRest) {
    bump('sh_rest', g.shRest.subarray(i * probe.n_rest, (i + 1) * probe.n_rest), c.sh_rest);
  }
}
console.log(`  ${'字段'.padEnd(10)}${'最大绝对误差'.padStart(14)}   结果`);
console.log('  ' + '-'.repeat(40));
let maxErr = 0;
for (const [k, v] of Object.entries(fields)) {
  if (v > maxErr) maxErr = v;
  const good = v < 1e-4;
  console.log(`  ${k.padEnd(10)}${v.toExponential(2).padStart(14)}   ${good ? 'OK' : '**FAIL**'}`);
}
const badFields = Object.entries(fields).filter(([, v]) => v >= 1e-4).map(([k]) => k);
const ok = badFields.length === 0;
chk(ok, ok ? `全部字段与 NumPy 一致（最大 ${maxErr.toExponential(2)}）`
           : `不一致字段：${badFields.join(', ')}（最大 ${maxErr.toExponential(2)}）`);

if (!ok) {
  // 打印首个样本的原始值，便于人工判断是"数值错"还是"期望值口径错"
  const c = probe.cases[0];
  console.log('\n=== 首个样本对照（JS | NumPy）===');
  console.log('  scale   JS', Array.from(g.scale.subarray(0, 3)).map((v) => v.toFixed(4)).join(', '),
    '| NP', c.scale.map((v) => v.toFixed(4)).join(', '));
  console.log('  opacity JS', g.opacity[0].toFixed(4), '| NP', c.opacity.toFixed(4));
  console.log('  sh_dc   JS', Array.from(g.shDC.subarray(0, 3)).map((v) => v.toFixed(4)).join(', '),
    '| NP', c.sh_dc.map((v) => v.toFixed(4)).join(', '));
}

// 参数维度统计：证明"信息没丢"
const dims = 3 + 3 + 4 + 1 + 3 + (probe.n_rest || 0);
console.log(`\n  每高斯参数维度 = 3(位置)+3(尺度)+4(旋转)+1(不透明度)+3(SH DC)+${probe.n_rest}(SH 高阶) = ${dims}`);
console.log('  对照：.splat = 11 维（丢 SH）；glTF 点云 = 6 维（丢 SH+尺度+旋转+不透明度）');

console.log(`\n结论: ${bad === 0 ? `完整 ${dims} 维全部读出，信息维度未丢` : `**${bad} 项不符**`}`);
process.exit(bad === 0 ? 0 : 1);
