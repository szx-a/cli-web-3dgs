/* 诊断"模型不在画面中央 / 在相机后面"的问题。
 *
 * 用户的观察：模型渲染位置不在查看器正中，跑到了视角后面导致不可见。
 * 这类问题**可以纯数值验证**：把模型包围盒的 8 个角点用查看器**同一套**
 * 相机数学（qRot / buildCam / eyePos / m4viewFromQuat / m4perspective）
 * 投影到屏幕空间，看：
 *   - 角点是否落在视锥内（|ndc| <= 1）
 *   - 在相机前方还是后方（view 空间 z 的符号）
 *   - 屏幕坐标是否居中
 *
 * 用法: node scripts/probe_viewer_framing_model.mjs <model.glb|ply>
 */
import fs from 'node:fs';
import path from 'node:path';

const file = process.argv[2];
if (!file) { console.error('用法: node scripts/probe_viewer_framing_model.mjs <model>'); process.exit(2); }

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
  const noop = () => {};
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
    createTexture: () => ({}), deleteTexture: noop, drawArrays: noop, drawElements: noop,
    getError: () => 0, getParameter: () => 4096,
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

// 一次性抽出"相机数学"整块源码。
// ⚠️ 顺序很反直觉：m4mul 在**行 28**、m4viewFromQuat 在 85、而 sub/dot/cross/norm 在 120+。
//    所以要从 `function m4mul(` 一直抽到 `norm` 的结束，才能拿到全部依赖
//    （第一版从 `const sub =` 开始，得到 0 字节，反复报 m4viewFromQuat is not defined）。
// 整块抽取天然保持源码顺序与依赖，与浏览器执行的是同一份代码。
const a1 = src.indexOf('function m4mul(');
const a2 = src.indexOf('function norm(');
if (a1 < 0 || a2 < 0) throw new Error('找不到相机数学区段');
let end = src.indexOf('{', a2);
let depth = 0;
for (let k = end; k < src.length; k++) {
  if (src[k] === '{') depth++;
  else if (src[k] === '}') { depth--; if (depth === 0) { end = k + 1; break; } }
}
const helperSrc = src.slice(a1, end);
const decls = (helperSrc.match(/(?:function|const)\s+[A-Za-z_$][\w$]*/g) || [])
  .map((s) => s.replace(/^(?:function|const)\s+/, ''));
console.log(`相机数学区段 ${helperSrc.length} 字节: ${decls.join(', ')}`);
for (const need of ['m4mul', 'qMul', 'qRot', 'm4viewFromQuat', 'm4perspective', 'norm', 'sub']) {
  if (!decls.includes(need)) throw new Error('区段里缺少 ' + need);
}

new Function('window', 'document', 'fetch', 'Blob', 'URL', 'Image', 'navigator',
             'getComputedStyle', src)(
  windowStub, globalThis.document, globalThis.fetch, globalThis.Blob, globalThis.URL,
  globalThis.Image, globalThis.navigator, globalThis.getComputedStyle);
const V = windowStub.GsViewer;
V.init(canvas, () => {});
const kind = file.toLowerCase().endsWith('.ply') ? 'ply' : 'glb';
const info = await V.loadBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
                                path.basename(file), kind);
const d = V.debug();
console.log(`模型: ${path.basename(file)}  顶点 ${info.vertices}  三角面 ${info.triangles} mode ${info.mode}`);
console.log(`相机: eye=[${d.camera.eye.map((v) => v.toFixed(3))}]  target=[${d.camera.target.map((v) => v.toFixed(3))}]  dist=${d.camera.dist.toFixed(3)}  mode=${d.camera.rotMode}`);

// 用同一套数学重算 view/proj，把包围盒角点投影到屏幕
const { qcam, dist, target, eye } = d.camera;
void qcam;
const aspect = 800 / 600;
// 先验证视图矩阵本身：把 eye 变换过去应当是原点，target 应当在 -Z 上
const viewCheck = new Function(helperSrc + '\nreturn m4viewFromQuat;')()(d.camera.quat, eye, target);
const xform = (m, v) => [
  m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12],
  m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13],
  m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14],
  m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15],
];
const eyeInView = xform(viewCheck, eye);
const tgtInView = xform(viewCheck, target);
console.log(`视图矩阵自检: eye->[${eyeInView.map((v) => v.toFixed(3))}]（应为 [0,0,0]）  ` +
            `target->[${tgtInView.map((v) => v.toFixed(3))}]（应为 [0,0,-dist]）`);

const view = viewCheck;
const proj = new Function(helperSrc + '\nreturn m4perspective;')()(Math.PI / 4, aspect, dist * 1e-3, dist * 100);
const mvp = new Function(helperSrc + '\nreturn m4mul;')()(proj, view);

// 包围盒：从顶点里取（debug 里没有 min/max，这里用 obj.center/radius 近似成球）
const c = d.obj.center, r = d.obj.radius;
console.log(`模型中心=[${c.map((v) => v.toFixed(3))}]  半径=${r.toFixed(3)}`);
const corners = [];
for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
  corners.push([c[0] + sx * r, c[1] + sy * r, c[2] + sz * r]);
}
const apply = (m, v) => {
  const x = m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12];
  const y = m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13];
  const z = m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14];
  const w = m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15];
  return [x / w, y / w, z / w];
};
let inside = 0, behind = 0;
const ndcs = [];
for (const p of corners) {
  const n = apply(mvp, p);
  ndcs.push(n);
  if (n[2] < -1 || n[2] > 1) behind++;              // 超出近远裁剪面
  if (Math.abs(n[0]) <= 1 && Math.abs(n[1]) <= 1 && n[2] >= -1 && n[2] <= 1) inside++;
}
const xs = ndcs.map((n) => n[0]), ys = ndcs.map((n) => n[1]), zs = ndcs.map((n) => n[2]);
console.log(`包围盒 8 角点 NDC 范围: x[${Math.min(...xs).toFixed(2)}, ${Math.max(...xs).toFixed(2)}]  ` +
            `y[${Math.min(...ys).toFixed(2)}, ${Math.max(...ys).toFixed(2)}]  z[${Math.min(...zs).toFixed(2)}, ${Math.max(...zs).toFixed(2)}]`);
const cndc = apply(mvp, c);
console.log(`模型中心的 NDC = [${cndc.map((v) => v.toFixed(3))}]  （应接近 [0,0,0]）`);
console.log(`视锥内角点 ${inside}/8`);
const centered = Math.abs(cndc[0]) < 0.15 && Math.abs(cndc[1]) < 0.15;
const visible = inside > 0;
console.log();
if (!visible) { console.log('**模型完全在视锥外 -> 不可见（用户所述现象）**'); }
else if (!centered) { console.log(`**模型中心偏离画面中心（NDC x=${cndc[0].toFixed(2)} y=${cndc[1].toFixed(2)}）**`); }
else { console.log('模型居中且可见'); }
process.exitCode = (visible && centered) ? 0 : 1;
