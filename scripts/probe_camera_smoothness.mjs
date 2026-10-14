/* 定量检查轨道相机的**主观"卡住"**问题（不只是"能否到两极"）。
 *
 * 为什么需要：四元数已解决"转不过极点"，但用户仍反馈"还有万向锁"。
 * 剩下的可能机制是**感觉上的卡住/反转**，常见的两个来源：
 *   A) 竖直拖动在**越过天顶后方向反转** —— 相机局部右轴翻向，继续向下拖变成向上转；
 *   B) 极点附近视图方向抖动（每像素变化量剧增）。
 * 这两者都不会产生 NaN，所以之前的探针测不出来，必须用"方向步长"来量化。
 *
 * 判据：
 *   1) 视线方向相邻步长的夹角应始终有限且不出现尖峰（无抖动/跳变）；
 *   2) 竖直拖动在**同一半球内**必须单调（向下拖 -> 相机向下走），
 *      即"方向自洽"，越界那一步允许反转（物理上必须）。
 *
 * 用法: node scripts/probe_camera_smoothness.mjs <model>
 */
import fs from 'node:fs';
import path from 'node:path';

const file = process.argv[2];
if (!file) { console.error('用法: node scripts/probe_camera_smoothness.mjs <model>'); process.exit(2); }

const listeners = { canvas: {}, window: {} };
function mkTarget(store) {
  return {
    addEventListener: (t, fn) => { (store[t] = store[t] || []).push(fn); },
    removeEventListener: () => {}, setPointerCapture: () => {},
  };
}
const GLV = {
  DEPTH_TEST: 0x0B71, COLOR_BUFFER_BIT: 0x4000, DEPTH_BUFFER_BIT: 0x0100,
  VERTEX_SHADER: 0x8B31, FRAGMENT_SHADER: 0x8B30, COMPILE_STATUS: 0x8B81,
  LINK_STATUS: 0x8B82, ARRAY_BUFFER: 0x8892, ELEMENT_ARRAY_BUFFER: 0x8893,
  STATIC_DRAW: 0x88E4, FLOAT: 0x1406, UNSIGNED_INT: 0x1405, UNSIGNED_BYTE: 0x1401,
  UNSIGNED_SHORT: 0x1403, RGBA: 0x1908, TEXTURE_2D: 0x0DE1, TEXTURE_WRAP_S: 0x2802,
  TEXTURE_WRAP_T: 0x2803, REPEAT: 0x2901, TEXTURE_MIN_FILTER: 0x2801,
  TEXTURE_MAG_FILTER: 0x2800, LINEAR_MIPMAP_LINEAR: 0x2703, LINEAR: 0x2601,
  UNPACK_FLIP_Y_WEBGL: 0x9240, POINTS: 0, TRIANGLES: 4, LINES: 1, TEXTURE0: 0x84C0,
  VENDOR: 0x1F00, RENDERER: 0x1F01, VERSION: 0x1F02,
};
function makeGl() {
  let err = 0; const noop = () => {};
  return {
    ...GLV,
    createShader: () => ({}), shaderSource: noop, compileShader: noop,
    getShaderParameter: () => true, getShaderInfoLog: () => '',
    createProgram: () => ({}), attachShader: noop, linkProgram: noop,
    getProgramParameter: () => true, getProgramInfoLog: () => '', useProgram: noop,
    getUniformLocation: (p, n) => ({ n }),
    getAttribLocation: (p, n) => ({ aPos: 0, aCol: 1, aUV: 2, aNrm: 3 })[n] ?? -1,
    createBuffer: () => ({}), bindBuffer: noop, bufferData: noop, deleteBuffer: noop,
    enableVertexAttribArray: noop, disableVertexAttribArray: noop,
    vertexAttribPointer: noop, vertexAttrib2f: noop, vertexAttrib3f: noop,
    enable: noop, disable: noop, depthFunc: noop, clearColor: noop, clear: noop,
    viewport: noop, uniformMatrix4fv: noop, uniform1f: noop, uniform1i: noop,
    uniform3f: noop, activeTexture: noop, bindTexture: noop, pixelStorei: noop,
    texImage2D: noop, texParameteri: noop, generateMipmap: noop,
    createTexture: () => ({}), deleteTexture: noop, drawArrays: noop,
    drawElements: (m, c, ty) => {
      if (ty !== GLV.UNSIGNED_SHORT && ty !== GLV.UNSIGNED_BYTE && ty !== GLV.UNSIGNED_INT) {
        err = err || 0x0500;
      }
    },
    getError: () => { const e = err; err = 0; return e; },
    getParameter: () => 4096,
    getExtension: (n) => (n === 'OES_element_index_uint' ? { n } : null),
    isContextLost: () => false,
  };
}
const canvas = Object.assign(mkTarget(listeners.canvas), {
  clientWidth: 800, clientHeight: 600, width: 800, height: 600, getContext: () => makeGl(),
});
const windowStub = Object.assign(mkTarget(listeners.window), { devicePixelRatio: 1 });
globalThis.window = windowStub;
globalThis.document = { getElementById: () => canvas };
globalThis.self = windowStub;
try { Object.defineProperty(globalThis, 'navigator', { value: { userAgent: 'node' }, configurable: true }); } catch {}
globalThis.getComputedStyle = () => ({ width: '800px', height: '600px', display: 'block', visibility: 'visible' });
const buf = fs.readFileSync(file);
globalThis.fetch = async () => ({
  ok: true, status: 200,
  arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
});
globalThis.Blob = class {}; globalThis.URL = { createObjectURL: () => 'blob:x' };
globalThis.Image = class { set src(_) { if (this.onerror) this.onerror(new Error('x')); } };
const src = fs.readFileSync(path.join('src', 'gsflow', 'web', 'static', 'viewer.js'), 'utf8');
new Function('window', 'document', 'fetch', 'Blob', 'URL', 'Image', 'navigator',
             'getComputedStyle', src)(
  windowStub, globalThis.document, globalThis.fetch, globalThis.Blob, globalThis.URL,
  globalThis.Image, globalThis.navigator, globalThis.getComputedStyle);
const V = windowStub.GsViewer;
V.init(canvas, () => {});
await V.loadBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
                    path.basename(file), file.toLowerCase().endsWith('.ply') ? 'ply' : 'glb');

const fire = (store, t, ev) => (store[t] || []).forEach((f) => f(ev));
/** 一次拖拽；返回拖后的单位视线方向（相机 -> 目标 的反向，即"相机在哪一侧"） */
function drag(dx, dy) {
  fire(listeners.canvas, 'mousedown', { button: 0, shiftKey: false, clientX: 400, clientY: 300, preventDefault() {} });
  fire(listeners.window, 'mousemove', { clientX: 400 + dx, clientY: 300 + dy });
  fire(listeners.window, 'mouseup', {});
  const d = V.debug();
  const e = d.camera.eye, t = d.camera.target;
  const v = [e[0] - t[0], e[1] - t[1], e[2] - t[2]];
  const L = Math.hypot(...v) || 1;
  return v.map((x) => x / L);
}
const ang = (a, b) => {
  const d = Math.min(1, Math.max(-1, a[0]*b[0] + a[1]*b[1] + a[2]*b[2]));
  return Math.acos(d);
};

let bad = 0;

/* ---- 测试 1：竖直拖动逐像素，检查步长均匀性与方向自洽 ---- */
V.reset();
const stepY = [];
let prev = drag(0, 0);
for (let i = 0; i < 900; i++) {          // 900px，每步 1px；总 7.2 rad，必然越过两极
  const cur = drag(0, 5);
  stepY.push({ d: ang(prev, cur), y: cur[1], prevY: prev[1] });
  prev = cur;
}
const ds = stepY.map((s) => s.d);
const dMean = ds.reduce((a, b) => a + b, 0) / ds.length;
const dMax = Math.max(...ds);
// 方向自洽：同一半球内，向下拖（dy=+5，即 camera 向下转）应使 y **下降**？
// 注意 orbit 里竖直量是 -dy，所以 dy=+5 让相机向"上"转：y 应上升。这里只检查
// **同一半球内方向一致**（不要求具体符号），跨极点的反向是允许的。
let inconsistent = 0;
for (let i = 1; i < stepY.length; i++) {
  const a = stepY[i - 1], b = stepY[i];
  const sameSide = Math.sign(a.prevY) === Math.sign(b.y) && Math.abs(a.prevY) > 0.05;
  if (sameSide && Math.sign(b.y - a.prevY) !== Math.sign(stepY[0].y - 0)) {
    // 允许符号与首次不同，只要"同一半球内变化方向一致"即可
  }
}
// ⚠️ 判据修正（我第一版写错了）：竖直连续拖会绕很多圈，y 分量本来就**像正弦一样
// 正负摆动**，所以"统计 y 变化方向的符号翻转次数"必然得到 ≈圈数，与"卡不卡"无关。
//
// 真正能区分两种实现的量是：**相机在不经过极点的前提下能连续走多远**。
//   - 正确实现（trackball）：从一极到另一极走满 180° 的弧；
//   - 缺陷实现（竖直绕相机右轴）：一旦越过分界角，拖动方向反转 ⇒
//     相机只走到 ±分界角就被"推回"，**永远无法跨越其余弧段**。
// 做法：把 900px 的竖直拖动切成小步，记录 y 的极值点；相邻极值点之间
// 视线方向的实际弧长应接近 π（180°），显著偏小就说明有卡住区。
const extremeIdx = [];
for (let i = 1; i < stepY.length - 1; i++) {
  const a = stepY[i - 1].y, b = stepY[i].y, c = stepY[i + 1].y;
  if ((b >= a && b >= c) || (b <= a && b <= c)) extremeIdx.push(i);
}
let arcMin = Infinity, arcMax = 0, arcs = [];
for (let k = 1; k < extremeIdx.length; k++) {
  // 极值点之间累加单步转角
  let arc = 0;
  for (let i = extremeIdx[k - 1] + 1; i <= extremeIdx[k]; i++) arc += stepY[i].d;
  if (arc > 0.3) { arcs.push(arc); arcMin = Math.min(arcMin, arc); arcMax = Math.max(arcMax, arc); }
}
const arcMean = arcs.length ? arcs.reduce((a, b) => a + b, 0) / arcs.length : 0;
// 竖直拖动时，相机在"同一段弧"内 y 应单调；用最大连续单调段长度衡量"能走多远"
let bestRun = 0, run = 0;
for (let i = 1; i < stepY.length; i++) {
  const same = Math.sign(stepY[i].y - stepY[i - 1].y) === Math.sign(stepY[i - 1].y - (stepY[i - 2] ? stepY[i - 2].y : stepY[i - 1].y));
  if (same) { run++; bestRun = Math.max(bestRun, run); } else { run = 0; }
}
console.log(`【竖直拖动 900px】单步（5px）视线转角：`);
console.log(`  平均 ${(dMean * 180 / Math.PI).toFixed(2)}°  最大 ${(dMax * 180 / Math.PI).toFixed(2)}°  `
            + `（最大/平均 = ${(dMax / dMean).toFixed(2)}）`);
console.log(`  极值点 ${extremeIdx.length} 个；相邻极值间弧长 平均 ${(arcMean * 180 / Math.PI).toFixed(1)}°  `
            + `最小 ${isFinite(arcMin) ? (arcMin * 180 / Math.PI).toFixed(1) : '-'}°  `
            + `最大 ${(arcMax * 180 / Math.PI).toFixed(1)}°`);
console.log(`  （理想应接近 180°；明显偏小 = 存在"推回"区段 = 主观卡住）`);
console.log(`  最长连续单调段 = ${bestRun} 步（越短越容易感到"转不动"）`);
// 抖动判据：单步转角**不应超过输入对应的旋转量**（5px × 0.008 = 2.29°）。
// ⚠️ 不能用"最大/平均"：相机顶到 ±90° 后步长归零，会把平均值拉低，使该比值虚高 ——
// 我第一版就是这么误判的（实测最大/平均=17.8，但最大步长恰好等于输入量，毫无抖动）。
const INPUT_DEG = 5 * 0.008 * 180 / Math.PI;   // 每次拖动对应的理论转角
console.log(`  理论单步 = ${INPUT_DEG.toFixed(2)}°（5px × 0.008 rad/px）`);
if (dMax > INPUT_DEG * 1.05) {
  console.log(`  **单步 ${(dMax * 180 / Math.PI).toFixed(2)}° 超过输入量 -> 有跳变**`);
  bad++;
}
if (arcMean > 0 && arcMean * 180 / Math.PI < 150) {
  console.log(`  **相邻极值间只能走 ${(arcMean * 180 / Math.PI).toFixed(1)}°，远小于 180° -> 有卡住区段**`);
  bad++;
}

/* ---- 测试 2：水平拖动是否始终有效（含极点处） ---- */
// ⚠️ 判据修正：**相机正对天顶/天底时，水平拖动确实不改变视线方向** ——
// 这是几何必然（视线与 yaw 轴共线），不是 bug。第一版在这里判"失效"是过严的，
// 会把正确实现误判为失败。改为：在**非极点**处水平拖动必须有效。
V.reset();
const beforeYaw = drag(0, 0);
const afterYaw = drag(60, 0);
const yawDeg = ang(beforeYaw, afterYaw) * 180 / Math.PI;
console.log(`\n【非极点处水平拖动 60px】视线转角 = ${yawDeg.toFixed(2)}°（应明显 >0）`);
if (yawDeg < 8) { console.log('  **水平拖动无效**'); bad++; }

// 极点处水平拖动：改为断言"yaw 仍在累积（相机基在转）"，而不是视线不动
for (let i = 0; i < 60; i++) drag(0, 40);     // 先顶到极点
const q1 = V.debug().camera.quat;
drag(80, 0);
const q2 = V.debug().camera.quat;
const qdot = Math.abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]);
console.log(`  极点处水平拖动后朝向四元数夹角 = ${(2 * Math.acos(Math.min(1, qdot)) * 180 / Math.PI).toFixed(2)}°`
            + `（>0 表示 yaw 生效，只是视线方向不变 —— 几何必然）`);
if (qdot > 0.99999) { console.log('  **极点处 yaw 完全没生效）'); bad++; }

/* ---- 测试 3：任意方向连续拖 200 次，全程步长有限 ---- */
V.reset();
let worst = 0, nanSeen = 0;
prev = drag(0, 0);
for (let i = 0; i < 200; i++) {
  const dx = ((i * 37) % 120) - 60, dy = ((i * 53) % 120) - 60;
  const cur = drag(dx, dy);
  if (!cur.every(Number.isFinite)) { nanSeen++; break; }
  worst = Math.max(worst, ang(prev, cur));
  prev = cur;
}
console.log(`\n【随机方向连续拖 200 次】最大单步转角 = ${(worst * 180 / Math.PI).toFixed(1)}°  非有限 = ${nanSeen}`);
if (nanSeen) { console.log('  **出现非有限方向**'); bad++; }

console.log('\n结论: ' + (bad ? `**发现 ${bad} 类问题**` : '相机旋转连续、无抖动、极点处仍可控'));
process.exitCode = bad ? 1 : 0;
