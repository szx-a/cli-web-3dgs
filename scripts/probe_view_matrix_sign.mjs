// 用**可控实例**验证 m4viewFromQuat 的符号约定（不看真实模型，避免被数据干扰）。
//
// 用户报告：模型不在画面中央、跑到视角后面导致不可见。
// 实测真实模型得到 target -> [0,0,+7.681]，而相机应当看向 -Z，可疑。
// 这里用最简单的已知答案来判定：
//   eye = [0,0,5], target = [0,0,0], 单位四元数（无旋转）
//   若视图矩阵正确，target 应变换到 [0,0,-5]；eye 应到 [0,0,0]。
//
// 用法: node scripts/probe_view_matrix_sign.mjs
import fs from 'node:fs';

const src = fs.readFileSync('src/gsflow/web/static/viewer.js', 'utf8');
const a1 = src.indexOf('function m4mul(');
const a2 = src.indexOf('function norm(');
let end = src.indexOf('{', a2);
let depth = 0;
for (let k = end; k < src.length; k++) {
  if (src[k] === '{') depth++;
  else if (src[k] === '}') { depth--; if (depth === 0) { end = k + 1; break; } }
}
const helperSrc = src.slice(a1, end);
const api = new Function(helperSrc + `
  return { m4viewFromQuat, m4perspective, m4mul, qRot, qAxisAngle, qNorm,
           m4lookAt, sub, norm, cross, dot };
`)();

const xform = (m, v) => [
  m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12],
  m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13],
  m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14],
  m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15],
];
const f3 = (v) => '[' + v.slice(0, 3).map((x) => x.toFixed(4)).join(', ') + ']';

const eye = [0, 0, 5], target = [0, 0, 0];
const ident = [1, 0, 0, 0];
const view = api.m4viewFromQuat(ident, eye, target);
console.log('m4viewFromQuat（单位四元数, eye=[0,0,5], target=[0,0,0]）');
console.log('  eye    -> ' + f3(xform(view, eye)) + '   期望 [0, 0, 0]');
console.log('  target -> ' + f3(xform(view, target)) + '   期望 [0, 0, -5]');
const t = xform(view, target);
const ok = Math.abs(t[0]) < 1e-6 && Math.abs(t[1]) < 1e-6 && t[2] < -4.9;
console.log(ok ? '  ✅ 符号约定正确（看向 -Z）' : '  ❌ **符号错误：目标被放在 +Z（相机背后）**');

console.log('\n对照 m4lookAt（同一输入）：');
const vl = api.m4lookAt(eye, target, [0, 1, 0]);
console.log('  eye    -> ' + f3(xform(vl, eye)) + '   期望 [0, 0, 0]');
console.log('  target -> ' + f3(xform(vl, target)) + '   期望 [0, 0, -5]');

console.log('\n用 perspective 验证完整链路（点应落在 NDC 内）：');
// far 必须明显大于相机到目标的距离，否则目标恰好落在远裁剪面上：
// 我第一版用 far=500 但**没除以 w**（把齐次坐标当 NDC 打印），
// 得到 -5.0101 而误判"视锥外"。这里两个问题都修掉。
const proj = api.m4perspective(Math.PI / 4, 4 / 3, 0.005, 500);
const ndcOf = (m, v) => { const r = xform(m, v); return [r[0] / r[3], r[1] / r[3], r[2] / r[3]]; };
const inFrustum = (n) => n[2] > -1 && n[2] < 1 && Math.abs(n[0]) <= 1 && Math.abs(n[1]) <= 1;
const ndc = ndcOf(api.m4mul(proj, api.m4viewFromQuat(ident, eye, target)), target);
console.log('  target 的 NDC = ' + f3(ndc) + (inFrustum(ndc) ? '  ✅ 在视锥内' : '  ❌ 在视锥外'));
const ndc2 = ndcOf(api.m4mul(proj, vl), target);
console.log('  （m4lookAt 版）NDC = ' + f3(ndc2) + (inFrustum(ndc2) ? '  ✅ 在视锥内' : '  ❌ 在视锥外'));
// 两者必须一致（同一相机朝向应当给出同样的投影）
const same = Math.abs(ndc[2] - ndc2[2]) < 1e-4;
console.log('  m4viewFromQuat 与 m4lookAt 的 NDC 是否一致 = ' + (same ? '是 ✅' : '否 ❌'));

process.exitCode = (ok && inFrustum(ndc) && same) ? 0 : 1;
