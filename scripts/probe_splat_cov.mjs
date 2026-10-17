/**
 * 交叉验证 `splat.js` 的解析与协方差计算（与 NumPy 的期望值逐元素比对）。
 *
 * 用法: node scripts/probe_splat_cov.mjs
 *
 * 为什么要它：泼溅渲染全依赖 3×3 协方差 M = R·diag(s²)·Rᵀ；
 * 四元数顺序、尺度是否平方、矩阵转置任何一处错，椭球朝向就会全错，
 * 而**在没有浏览器的环境里看不出来**。这里用独立实现（NumPy）算期望值做比对。
 */
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(import.meta.dirname, '..');
const probePath = path.join(ROOT, 'env', 'tmp', 'splat_probe.json');
if (!fs.existsSync(probePath)) {
  console.error('缺少 probe 文件，请先运行: python scripts/dump_splat_probe.py --job img3-crop');
  process.exit(2);
}
const probe = JSON.parse(fs.readFileSync(probePath, 'utf8'));

// splat.js 引用了 window，这里给个最小 shim 再 eval
const src = fs.readFileSync(path.join(ROOT, 'src', 'gsflow', 'web', 'static', 'splat.js'), 'utf8');
const shimWindow = {};
const fn = new Function('window', src + '\nreturn window.GsSplatRenderer;');
const S = fn(shimWindow);

// 1) 全量解析一遍，校验条数与字节数
const buf = fs.readFileSync(probe.splat_path);
const parsed = S.parseSplat(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
console.log('=== 解析 ===');
console.log(`  字节数 ${buf.byteLength} ⇒ count ${parsed.count}, 期望 ${probe.count}`);
console.log(`  条数匹配: ${parsed.count === probe.count ? 'OK' : '**FAIL**'}`);
console.log(`  字节数匹配: ${buf.byteLength === probe.bytes ? 'OK' : '**FAIL**'}`);

// 2) 逐样本比对位置与协方差
let bad = 0;
console.log('\n=== 协方差比对（JS vs NumPy）===');
console.log('  idx   最大相对误差   位置一致   协方差一致');
console.log('  ' + '-'.repeat(50));
for (const c of probe.cases) {
  const i = c.idx;
  const gotCov = parsed.cov.slice(i * 6, i * 6 + 6);
  const gotCen = parsed.center.slice(i * 3, i * 3 + 3);
  let maxRel = 0;
  for (let k = 0; k < 6; k++) {
    const e = c.expect_cov[k];
    const g = gotCov[k];
    const rel = Math.abs(e - g) / Math.max(Math.abs(e), 1e-12);
    if (rel > maxRel) maxRel = rel;
  }
  const posOk = gotCen.every((v, k) => Math.abs(v - c.expect_center[k]) < 1e-6);
  const covOk = maxRel < 1e-5;
  if (!posOk || !covOk) bad++;
  console.log(`  ${String(i).padStart(3)}   ${maxRel.toExponential(2).padStart(10)}`
    + `   ${posOk ? 'OK' : '**FAIL**'}       ${covOk ? 'OK' : '**FAIL**'}`);
}

// 3) 单一尺度检验：单位四元数 + 各向异性尺度 ⇒ 协方差应等于 diag(s²)
{
  const s = [2.0, 3.0, 5.0];
  const cov = S.covFromScaleQuat(s[0], s[1], s[2], 1, 0, 0, 0);
  const want = [4, 0, 0, 9, 0, 25];
  const ok = cov.every((v, k) => Math.abs(v - want[k]) < 1e-6);
  console.log(`\n=== 已知答案检验 ===`);
  console.log(`  单位四元数 + 尺度(2,3,5) ⇒ 期望 diag(4,9,25)，实际`
    + ` [${cov.map((v) => v.toFixed(3)).join(', ')}]  ${ok ? 'OK' : '**FAIL**'}`);
  if (!ok) bad++;
}

// 4) 90° 绕 Y 旋转：尺度 (1,1,10) 的"长轴"应转到 X 方向
{
  const q = [Math.SQRT1_2, 0, Math.SQRT1_2, 0];   // w,y
  const cov = S.covFromScaleQuat(1, 1, 10, q[0], q[1], q[2], q[3]);
  // 绕 Y 转 90° 后，原 +Z 长轴 → +X ⇒ xx ≈ 100, zz ≈ 1
  const ok = Math.abs(cov[0] - 100) < 1e-3 && Math.abs(cov[5] - 1) < 1e-3;
  console.log(`  绕 Y 转 90° + 尺度(1,1,10) ⇒ 期望 xx≈100, zz≈1，实际`
    + ` xx=${cov[0].toFixed(3)}, zz=${cov[5].toFixed(3)}  ${ok ? 'OK' : '**FAIL**'}`);
  if (!ok) bad++;
}

console.log(`\n结论: ${bad === 0 ? '全部一致（JS 实现与 NumPy 相符）' : `**${bad} 项不符**`}`);
process.exit(bad === 0 ? 0 : 1);
