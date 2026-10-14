/* 直接测用户的实际症状：**持续朝同一方向拖动，相机会不会停下或回弹？**
 *
 * 为什么单独做这个：前两个探针都不够准确 ——
 *   - "能否到两极"：两种实现都能过（大步长会跳过反转区）；
 *   - "相邻极值间弧长"：大步长同样能跳过，所以也分不出来。
 * 用户描述的是"转到某个方向就转不动了"，本质是**单调性问题**：
 * 只要持续向下拖，相机就应该持续往下走，直到越过天底。
 *
 * 判据：把拖动切成小步（模拟真实拖动），检查 y 分量是否**单调下降**。
 * 出现"下降→上升"的非极点翻转，就是那块"推回/卡住"区域。
 *
 * 用法: node scripts/probe_camera_monotonic.mjs <model>
 */
import fs from 'node:fs';
import path from 'node:path';

const file = process.argv[2];
if (!file) { console.error('用法: node scripts/probe_camera_monotonic.mjs <model>'); process.exit(2); }

const listeners = { canvas: {}, window: {} };
const mkTarget = (s) => ({
  addEventListener: (t, fn) => { (s[t] = s[t] || []).push(fn); },
  removeEventListener: () => {}, setPointerCapture: () => {},
});
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
    ...GLV, createShader: () => ({}), shaderSource: noop, compileShader: noop,
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

const fire = (s, t, ev) => (s[t] || []).forEach((f) => f(ev));
// 模拟**真实拖动**：按下之后连续移动（不重新按下），这才是用户的手感
fire(listeners.canvas, 'mousedown', { button: 0, shiftKey: false, clientX: 400, clientY: 0, preventDefault() {} });
let y = 0;
const ys = [];
for (let i = 0; i < 1200; i++) {          // 1200 步 × 2px = 2400px 向下拖
  y += 2;
  fire(listeners.window, 'mousemove', { clientX: 400, clientY: y });
  const d = V.debug();
  const e = d.camera.eye, t = d.camera.target;
  const v = [e[0] - t[0], e[1] - t[1], e[2] - t[2]];
  const L = Math.hypot(...v) || 1;
  ys.push(v[1] / L);
}
fire(listeners.window, 'mouseup', {});

// 单调性分析：向下拖时 y（相机高度）应持续下降。
// 统计"非极点区"的反向步数：|y|>0.15 时若 y 上升即为异常（下方极点是 -1，故下降=变负）。
let reversals = 0;
const revAt = [];
for (let i = 1; i < ys.length; i++) {
  const dy = ys[i] - ys[i - 1];
  const nearPole = Math.abs(ys[i]) > 0.98;
  // 向下拖 -> 相机向下走 -> y 变负。所以 y 上升（dy>0）且不在极点附近即为反向。
  if (dy > 0 && !nearPole) { reversals++; if (revAt.length < 6) revAt.push({ i, y: ys[i].toFixed(3) }); }
}
const yMin = Math.min(...ys), yMax = Math.max(...ys);
console.log('  y 序列采样（每 60 步）: ' + ys.filter((_,i)=>i%60===0).map(v=>v.toFixed(3)).join(' '));
console.log(`持续向下拖 2400px（1200 步 × 2px）：`);
console.log(`  y 范围 ${yMin.toFixed(3)} ~ ${yMax.toFixed(3)}`);
console.log(`  非极点区反向步数 = ${reversals}`);
if (revAt.length) console.log(`  前几个反向位置：` + revAt.map((r) => `step ${r.i}(y=${r.y})`).join(', '));
console.log(`  是否到达下方极点（y<=-0.98）= ${yMin <= -0.98 ? '是' : '否'}`);
if (reversals > 0) {
  console.log(`\n**存在 ${reversals} 步反向 -> 拖动中途被"推回"，主观就是卡住**`);
  process.exitCode = 1;
} else {
  console.log('\n结论: 持续拖动能单调穿越两极，无推回区段');
  process.exitCode = 0;
}
