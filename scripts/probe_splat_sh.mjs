/**
 * 交叉验证 `splat.js` 的 **SH 着色**（与 NumPy 的官方 `eval_sh` 实现比对）。
 *
 * 用法: python scripts/dump_sh_probe.py  →  node scripts/probe_splat_sh.mjs
 *
 * ## 为什么必须验证
 *
 * "完整实现信息维度不要丢"的核心就是**不能把 SH 丢掉**：
 * 完整 3DGS 每个高斯 59 个参数（位置3 + 尺度3 + 旋转4 + 不透明度1 + SH 48），
 * 其中 48 维 SH 决定**视角相关颜色**。若 SH 求值的常数/符号/基索引写错，
 * 颜色会整体偏移或方向反了，而**在没有浏览器时完全看不出来**。
 *
 * 这里用 NumPy 按官方 `sh_utils.py::eval_sh` 独立算一遍期望值，
 * 与 JS 的 `evalSHInto` 逐通道比对。
 */
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(import.meta.dirname, '..');
const probePath = path.join(ROOT, 'env', 'tmp', 'sh_probe.json');
if (!fs.existsSync(probePath)) {
  console.error('缺少 probe 文件，请先运行: python scripts/dump_sh_probe.py');
  process.exit(2);
}
const probe = JSON.parse(fs.readFileSync(probePath, 'utf8'));

const src = fs.readFileSync(path.join(ROOT, 'src', 'gsflow', 'web', 'static', 'splat.js'), 'utf8');
const S = new Function('window', src + '\nreturn window.GsSplatRenderer;')({});

// 把样本拼成 evalSHInto 需要的扁平数组（SH 布局：每高斯 48 = 3 通道 × 16 基）
const n = probe.cases.length;
const center = new Float32Array(n * 3);
const sh = new Float32Array(n * 48);
probe.cases.forEach((c, i) => {
  center[i * 3] = c.xyz[0];
  center[i * 3 + 1] = c.xyz[1];
  center[i * 3 + 2] = c.xyz[2];
  // c.shs 是 (16,3)：基索引 × 通道
  for (let b = 0; b < 16; b++) {
    for (let ch = 0; ch < 3; ch++) {
      sh[i * 48 + ch * 16 + b] = c.shs[b][ch];
    }
  }
});

let bad = 0;
console.log('=== SH 着色比对（JS vs NumPy 官方 eval_sh）===');
console.log('  视线            样本  最大绝对误差   结果');
console.log('  ' + '-'.repeat(52));
for (const eye of probe.eyes) {
  const color = new Float32Array(n * 4);
  color.fill(0);
  // 先把 alpha 设 1，便于只看 RGB
  for (let i = 0; i < n; i++) color[i * 4 + 3] = 1;
  S.evalSHInto(sh, probe.sh_degree, center, n, eye, color);
  let maxErr = 0;
  for (let i = 0; i < n; i++) {
    const want = probe.cases[i].expect_rgb[probe.eyes.indexOf(eye)];
    for (let ch = 0; ch < 3; ch++) {
      const e = Math.abs(color[i * 4 + ch] - want[ch]);
      if (e > maxErr) maxErr = e;
    }
  }
  const ok = maxErr < 2e-5;
  if (!ok) bad++;
  console.log(`  [${eye.map((v) => v.toFixed(1)).join(',')}]`.padEnd(17)
    + `${String(n).padStart(4)}  ${maxErr.toExponential(2).padStart(12)}   ${ok ? 'OK' : '**FAIL**'}`);
}

// 已知答案：deg=0 时颜色应恒为 0.5 + C0·f_dc，与视线无关
{
  const c1 = new Float32Array([0, 0, 0, 1]);
  const cen1 = new Float32Array([1, 2, 3]);
  const sh1 = new Float32Array(48);
  sh1[0] = 1.0;                     // R 通道基 0
  const e1 = [10, 0, 0], e2 = [-10, 0, 0];
  const a = new Float32Array(4), b = new Float32Array(4);
  S.evalSHInto(sh1, 0, cen1, 1, e1, a);
  S.evalSHInto(sh1, 0, cen1, 1, e2, b);
  const want = 0.5 + 0.28209479177387814 * 1.0;
  const ok = Math.abs(a[0] - want) < 1e-6 && Math.abs(b[0] - want) < 1e-6
    && Math.abs(a[0] - b[0]) < 1e-9;
  console.log(`\n=== 已知答案检验 ===`);
  console.log(`  deg=0, f_dc_R=1 ⇒ 期望 R=${want.toFixed(6)}，实际 ${a[0].toFixed(6)}；`
    + `两次视线应完全相同 ${Math.abs(a[0] - b[0]) < 1e-9 ? 'OK' : '**FAIL**'}  ${ok ? 'OK' : '**FAIL**'}`);
  if (!ok) bad++;
}

// 已知答案：deg=3 时改变视线方向，颜色**必须变化**（证明 SH 真的生效）
{
  const cen1 = new Float32Array([0, 0, 0]);
  const sh1 = new Float32Array(48);
  sh1[16 + 9] = 0.5;                // G 通道、基 9，让方向依赖明显
  const a = new Float32Array([0, 0, 0, 1]);
  const b = new Float32Array([0, 0, 0, 1]);
  S.evalSHInto(sh1, 3, cen1, 1, [0, 10, 0], a);
  S.evalSHInto(sh1, 3, cen1, 1, [10, 0, 0], b);
  const diff = Math.abs(a[1] - b[1]);
  const ok = diff > 1e-3;
  console.log(`  deg=3 且基 9 非零 ⇒ 换视线后 G 通道应变化，实测差 ${diff.toExponential(2)}  ${ok ? 'OK' : '**FAIL**'}`);
  if (!ok) bad++;
}

console.log(`\n结论: ${bad === 0 ? 'SH 求值与 NumPy 一致（信息维度未丢）' : `**${bad} 项不符**`}`);
process.exit(bad === 0 ? 0 : 1);
