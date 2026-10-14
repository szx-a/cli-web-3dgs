/* 验证两种相机旋转模式都可切换且各自符合定义。
 *
 *   clamp（默认）—— 持续同向拖动应**单调**（反向 0 步），到 ±90° 停住；
 *   free         —— 可**翻越极点**，视野方向能到达两个极点。
 *
 * 为什么要两种：用户三次反馈"卡住"，而"到极点停住"（标准轨道相机）与
 * "翻越极点"（自由翻转）是两种都合理但互不兼容的手感，不该靠猜。
 *
 * 用法: node scripts/probe_camera_modes.mjs <model>
 */
import fs from 'node:fs';
import path from 'node:path';

const file = process.argv[2];
if (!file) { console.error('用法: node scripts/probe_camera_modes.mjs <model>'); process.exit(2); }

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
function dragDown(steps, px) {
  fire(listeners.canvas, 'mousedown', { button: 0, shiftKey: false, clientX: 400, clientY: 0, preventDefault() {} });
  let y = 0; const ys = [];
  for (let i = 0; i < steps; i++) {
    y += px;
    fire(listeners.window, 'mousemove', { clientX: 400, clientY: y });
    const d = V.debug();
    const e = d.camera.eye, t = d.camera.target;
    const v = [e[0] - t[0], e[1] - t[1], e[2] - t[2]];
    const L = Math.hypot(...v) || 1;
    ys.push(v[1] / L);
  }
  fire(listeners.window, 'mouseup', {});
  return ys;
}
function reversals(ys) {
  let n = 0;
  for (let i = 1; i < ys.length; i++) {
    if (ys[i] - ys[i - 1] > 0 && Math.abs(ys[i]) <= 0.98) n++;
  }
  return n;
}

let bad = 0;
const def = V.getRotMode ? V.getRotMode() : '(不支持 getRotMode)';
console.log(`默认模式 = ${def}`);
if (def !== 'clamp') { console.log('  **默认模式不是 clamp**'); bad++; }
if (typeof V.setRotMode !== 'function') { console.log('  **缺少 setRotMode**'); bad++; }

// clamp：单调到极点
V.setRotMode('clamp'); V.reset();
const yc = dragDown(1200, 2);
const rc = reversals(yc);
console.log(`clamp 反向步数 = ${rc}   y 范围 ${Math.min(...yc).toFixed(3)} ~ ${Math.max(...yc).toFixed(3)}`);
if (rc !== 0) { console.log('  **clamp 模式出现反向**'); bad++; }

// free：能翻越到两个极点
V.setRotMode('free'); V.reset();
const yf = dragDown(1400, 2);
console.log(`free  y 范围 ${Math.min(...yf).toFixed(3)} ~ ${Math.max(...yf).toFixed(3)}`);
const bothPoles = Math.min(...yf) <= -0.98 && Math.max(...yf) >= 0.98;
console.log(`free 能到达两个极点 = ${bothPoles ? '是' : '否'}`);
if (!bothPoles) { console.log('  **free 模式无法翻越极点**'); bad++; }

V.setRotMode('clamp');   // 复原
console.log('\n结论: ' + (bad ? `**有 ${bad} 项问题**` : '两种模式均符合定义且可切换'));
process.exitCode = bad ? 1 : 0;
