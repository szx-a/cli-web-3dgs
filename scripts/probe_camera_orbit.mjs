/* 验证四元数轨道相机**能越过天顶/天底**（万向锁的真实症状）。
 *
 * 背景：旧实现用欧拉角 pitch，必须夹在 ±(π/2−ε)，于是"拖到顶/底就过不去"。
 * 用户三次反馈"还是有万向锁"。改成四元数后应当可以自由翻越。
 *
 * 判据：持续向下拖，记录相机方向（eye - target 归一化）的 y 分量：
 *   - 若能越过天顶：y 先升到 ≈+1，**然后下降**（从另一侧下来）
 *   - 若仍被夹取：y 会停在 ≈+1 不再变化
 * 同时全程 uMVP 必须有限（无 NaN）。
 *
 * 用法: node scripts/probe_camera_orbit.mjs <file.glb|ply>
 */
import fs from 'node:fs';
import path from 'node:path';

const file = process.argv[2];
if (!file) { console.error('用法: node scripts/probe_camera_orbit.mjs <model>'); process.exit(2); }

const listeners = { canvas: {}, window: {} };
function mkTarget(store) {
  return {
    addEventListener: (t, fn) => { (store[t] = store[t] || []).push(fn); },
    removeEventListener: () => {}, setPointerCapture: () => {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
  };
}
const GLV = {
  DEPTH_TEST: 0x0B71, COLOR_BUFFER_BIT: 0x4000, DEPTH_BUFFER_BIT: 0x0100,
  VERTEX_SHADER: 0x8B31, FRAGMENT_SHADER: 0x8B30,
  COMPILE_STATUS: 0x8B81, LINK_STATUS: 0x8B82, VALIDATE_STATUS: 0x8B83,
  ARRAY_BUFFER: 0x8892, ELEMENT_ARRAY_BUFFER: 0x8893, STATIC_DRAW: 0x88E4,
  FLOAT: 0x1406, UNSIGNED_INT: 0x1405, UNSIGNED_BYTE: 0x1401, UNSIGNED_SHORT: 0x1403,
  RGBA: 0x1908, TEXTURE_2D: 0x0DE1, TEXTURE_WRAP_S: 0x2802, TEXTURE_WRAP_T: 0x2803,
  REPEAT: 0x2901, TEXTURE_MIN_FILTER: 0x2801, TEXTURE_MAG_FILTER: 0x2800,
  LINEAR_MIPMAP_LINEAR: 0x2703, LINEAR: 0x2601,
  UNPACK_FLIP_Y_WEBGL: 0x9240, POINTS: 0, TRIANGLES: 4, LINES: 1, TEXTURE0: 0x84C0,
  VENDOR: 0x1F00, RENDERER: 0x1F01, VERSION: 0x1F02,
};
let mvpFinite = [];
let useUint = true;

function makeGl() {
  let err = 0;
  const noop = () => {};
  const g = {
    ...GLV,
    createShader: () => ({ id: 's' }), shaderSource: noop, compileShader: noop,
    getShaderParameter: () => true, getShaderInfoLog: () => '',
    createProgram: () => ({ id: 'p' }), attachShader: noop, linkProgram: noop,
    getProgramParameter: () => true, getProgramInfoLog: () => '',
    useProgram: noop,
    getUniformLocation: (p, n) => ({ n }),
    getAttribLocation: (p, n) => ({ aPos: 0, aCol: 1, aUV: 2, aNrm: 3 })[n] ?? -1,
    createBuffer: () => ({ id: 'b' }), bindBuffer: noop, bufferData: noop,
    deleteBuffer: noop, enableVertexAttribArray: noop, disableVertexAttribArray: noop,
    vertexAttribPointer: noop, vertexAttrib2f: noop, vertexAttrib3f: noop,
    enable: noop, disable: noop, depthFunc: noop, clearColor: noop, clear: noop,
    viewport: noop,
    uniformMatrix4fv: (l, tr, m) => {
      if (l && l.n === 'uMVP' && m) mvpFinite.push(Array.from(m).every(Number.isFinite));
    },
    uniform1f: noop, uniform1i: noop, uniform3f: noop,
    activeTexture: noop, bindTexture: noop, pixelStorei: noop,
    texImage2D: noop, texParameteri: noop, generateMipmap: noop,
    createTexture: () => ({ id: 't' }), deleteTexture: noop,
    drawArrays: noop,
    drawElements: (m, c, ty) => {
      if (ty === GLV.UNSIGNED_INT && !useUint) err = err || 0x0500;
      if (ty !== GLV.UNSIGNED_SHORT && ty !== GLV.UNSIGNED_BYTE && ty !== GLV.UNSIGNED_INT) {
        err = err || 0x0500;
      }
    },
    getError: () => { const e = err; err = 0; return e; },
    getParameter: () => 4096,
    getExtension: (n) => (n === 'OES_element_index_uint' && useUint ? { n } : null),
    isContextLost: () => false,
  };
  return g;
}

const canvas = Object.assign(mkTarget(listeners.canvas), {
  clientWidth: 800, clientHeight: 600, width: 800, height: 600,
  getContext: () => makeGl(),
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
globalThis.Blob = class { constructor() {} };
globalThis.URL = { createObjectURL: () => 'blob:x' };
globalThis.Image = class { set src(_) { if (this.onerror) this.onerror(new Error('x')); } };

const src = fs.readFileSync(path.join('src', 'gsflow', 'web', 'static', 'viewer.js'), 'utf8');
new Function('window', 'document', 'fetch', 'Blob', 'URL', 'Image', 'navigator', 'getComputedStyle',
             src)(windowStub, globalThis.document, globalThis.fetch, globalThis.Blob,
                  globalThis.URL, globalThis.Image, globalThis.navigator, globalThis.getComputedStyle);
const V = windowStub.GsViewer;
V.init(canvas, () => {});

const kind = file.toLowerCase().endsWith('.ply') ? 'ply' : 'glb';
await V.loadBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
                    path.basename(file), kind);

function fire(store, type, ev) { (store[type] || []).forEach((f) => f(ev)); }
/** 模拟一次"向下拖 dy 像素" */
function dragDown(dy) {
  fire(listeners.canvas, 'mousedown', { button: 0, shiftKey: false, clientX: 400, clientY: 300, preventDefault() {} });
  fire(listeners.window, 'mousemove', { clientX: 400, clientY: 300 + dy });
  fire(listeners.window, 'mouseup', {});
}

const ys = [];
const N = 60;             // 60 次，每次 200px；0.008 rad/px -> 每次 1.6 rad，足以翻越
for (let i = 0; i < N; i++) {
  dragDown(200);
  const d = V.debug();
  const eye = d.camera.eye, tg = d.camera.target;
  const v = [eye[0] - tg[0], eye[1] - tg[1], eye[2] - tg[2]];
  const L = Math.hypot(...v) || 1;
  ys.push(v[1] / L);
}

// 判据：**相机方向能同时到达两个极点**（y 接近 +1 与 −1），且全程无 NaN。
//
// 为什么不用"先升到峰值再回落"：每次拖动 1.6 rad，60 次会绕转很多圈，
// y 序列本来就是来回穿越两极的振荡（实测 0.89 → −0.59 → … → 1.00 → … → −0.98）。
// 旧实现（pitch 夹取）的 y 会**停在 ≈+1 附近不再变化**，所以"是否同时到达两极"
// 才是真正区分两者的判据。
const yMax = Math.max(...ys), yMin = Math.min(...ys);
// 另外统计"穿越"次数：相邻样本跨越 ±0.9 说明确实翻越了极点而不是停在附近
let crossings = 0;
for (let i = 1; i < ys.length; i++) {
  if ((ys[i - 1] > 0.9 && ys[i] < -0.9) || (ys[i - 1] < -0.9 && ys[i] > 0.9)) crossings++;
}
const reachedBoth = yMax > 0.9 && yMin < -0.9;
const nanCount = mvpFinite.filter((x) => !x).length;

console.log(`向下拖 ${N} 次（每次 200px），相机方向 y 分量：`);
console.log(`  y 范围 ${yMin.toFixed(3)} ~ ${yMax.toFixed(3)}（起始 ${ys[0].toFixed(3)}）`);
console.log(`  uMVP 采样 ${mvpFinite.length} 次，含 NaN/Inf 的 = ${nanCount}`);
console.log(`  是否同时到达两极 = ${reachedBoth ? '是' : '否'}；跨越极点次数 = ${crossings}`);

if (nanCount > 0) { console.log('**出现 NaN 矩阵**'); process.exitCode = 1; }
else if (!reachedBoth) {
  console.log('**相机只能到达一个极点（万向锁仍在）**');
  console.log('  y 序列（每 5 次取一个）：' + ys.filter((_, i) => i % 5 === 0).map((v) => v.toFixed(2)).join(' '));
  process.exitCode = 1;
} else {
  console.log('  => 相机可达天顶与天底、可连续翻越，四元数方案生效');
  process.exitCode = 0;
}
