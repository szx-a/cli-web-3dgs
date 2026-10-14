/* 在 Node 里加载**真实的 app.js + viewer.js**，用较真实的 DOM stub 跑一遍
 * 「模型查看」页的初始化，捕获任何异常。
 *
 * 为什么需要：用户在 Edge 里看到「画布全黑 + 状态栏无文字」。
 * 状态栏由 `GsViewer.init()` 的回调写入 —— 它为空说明**初始化没成功**。
 * 而之前所有排查都只测了 viewer.js 的解析/绘制，**从没跑过 app.js 的初始化**。
 *
 * 用法: node scripts/probe_gui_init.mjs
 */
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve('.');
const STATIC = path.join(ROOT, 'src', 'gsflow', 'web', 'static');

// ---- 较真实的 DOM stub：每个 id 一个独立元素 ----
class El {
  constructor(id = '', tag = 'div') {
    this.id = id; this.tagName = tag.toUpperCase();
    this.children = []; this.style = {}; this.dataset = {};
    this._html = ''; this._text = ''; this.value = '';
    this.checked = false; this.disabled = false; this.files = null;
    this.listeners = {};
    this.classList = {
      _s: new Set(),
      add: (...c) => c.forEach((x) => this.classList._s.add(x)),
      remove: (...c) => c.forEach((x) => this.classList._s.delete(x)),
      contains: (c) => this.classList._s.has(c),
      toggle: (c) => (this.classList._s.has(c) ? this.classList.delete(c) : this.classList.add(c)),
    };
    this.clientWidth = 800; this.clientHeight = 600;
    this.width = 800; this.height = 600;
  }
  get innerHTML() { return this._html; }
  set innerHTML(v) { this._html = String(v); this.children = []; }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); }
  appendChild(c) { this.children.push(c); return c; }
  removeChild(c) { this.children = this.children.filter((x) => x !== c); return c; }
  addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); }
  removeEventListener() {}
  setAttribute(k, v) { this[k] = v; }
  getAttribute(k) { return this[k]; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  getContext(kind) {
    if (this._gl) return this._gl;
    this._gl = makeGl(this);
    return this._gl;
  }
  getBoundingClientRect() { return { left: 0, top: 0, width: 800, height: 600 }; }
  focus() {} blur() {} click() {}
  setPointerCapture() {}
}

// 极简 GL stub：只要能跑通调用、并记录错误
function makeGl(el) {
  let err = 0;
  const noop = () => {};
  const gl = {
    DEPTH_TEST: 0x0B71, COLOR_BUFFER_BIT: 0x4000, DEPTH_BUFFER_BIT: 0x0100,
    VERTEX_SHADER: 0x8B31, FRAGMENT_SHADER: 0x8B30,
    COMPILE_STATUS: 0x8B81, LINK_STATUS: 0x8B82, VALIDATE_STATUS: 0x8B83,
    ARRAY_BUFFER: 0x8892, ELEMENT_ARRAY_BUFFER: 0x8893, STATIC_DRAW: 0x88E4,
    FLOAT: 0x1406, UNSIGNED_INT: 0x1405, UNSIGNED_BYTE: 0x1401, RGBA: 0x1908,
    TEXTURE_2D: 0x0DE1, TEXTURE_WRAP_S: 0x2802, TEXTURE_WRAP_T: 0x2803,
    REPEAT: 0x2901, TEXTURE_MIN_FILTER: 0x2801, TEXTURE_MAG_FILTER: 0x2800,
    LINEAR_MIPMAP_LINEAR: 0x2703, LINEAR: 0x2601, UNPACK_FLIP_Y_WEBGL: 0x9240,
    POINTS: 0, TRIANGLES: 4, LINES: 1, TEXTURE0: 0x84C0,
    VENDOR: 0x1F00, RENDERER: 0x1F01, VERSION: 0x1F02,
    NO_ERROR: 0, INVALID_ENUM: 0x0500, INVALID_VALUE: 0x0501,
    INVALID_OPERATION: 0x0502, CONTEXT_LOST_WEBGL: 0x9242, LESS: 0x0201,
    createShader: () => ({ id: 'sh' }),
    shaderSource: noop, compileShader: noop,
    getShaderParameter: (s, p) => {
      const ok = [0x8B4F, 0x8B80, 0x8B81];
      if (!ok.includes(p)) { err = err || 0x0500; return null; }
      return true;
    },
    getShaderInfoLog: () => '',
    createProgram: () => ({ id: 'p' }),
    attachShader: noop, linkProgram: noop,
    getProgramParameter: (p, pn) => {
      const ok = [0x8B80, 0x8B82, 0x8B83, 0x8B84, 0x8B86, 0x8B89, 0x8B8D];
      if (!ok.includes(pn)) { err = err || 0x0500; return null; }
      return true;
    },
    getProgramInfoLog: () => '', useProgram: noop,
    getUniformLocation: (p, n) => ({ n }),
    getAttribLocation: (p, n) => ({ aPos: 0, aCol: 1, aUV: 2, aNrm: 3 })[n] ?? -1,
    createBuffer: () => ({ id: 'b' }), bindBuffer: noop, bufferData: noop,
    deleteBuffer: noop, enableVertexAttribArray: noop, disableVertexAttribArray: noop,
    vertexAttribPointer: noop, vertexAttrib2f: noop, vertexAttrib3f: noop,
    enable: noop, disable: noop, depthFunc: noop, clearColor: noop, clear: noop,
    viewport: noop, uniformMatrix4fv: noop, uniform1f: noop, uniform1i: noop,
    uniform3f: noop, activeTexture: noop, bindTexture: noop, pixelStorei: noop,
    texImage2D: noop, texParameteri: noop, generateMipmap: noop,
    createTexture: () => ({ id: 't' }), deleteTexture: noop,
    drawArrays: noop, drawElements: noop,
    getError: () => { const e = err; err = 0; return e; },
    getParameter: (pn) => {
      const ok = [0x1F00, 0x1F01, 0x1F02, 0x0D33, 0x80A8, 0x80A9, 0x0B74,
                  0x8B8D, 0x0BA2, 0x0D3A, 0x84E0];
      if (!ok.includes(pn)) { err = err || 0x0500; return null; }
      return pn === 0x1F00 ? 'MockVendor' : (pn === 0x0D3A ? new Int32Array([16384, 16384]) : 4096);
    },
    isContextLost: () => false,
    // 真实 WebGL 一定有 getExtension；mock 必须提供，否则 createRenderer
    // 里的扩展探测会抛 "gl.getExtension is not a function"，
    // 把"探针不完整"误报成"产品缺陷"（实测踩过）。
    getExtension: (name) => (name === 'OES_element_index_uint' ? { name } : null),
  };
  // 把方法放到**原型**上，模拟真实 WebGL：own property 可写、可被包装
  const proto = Object.getPrototypeOf(gl);
  return Object.assign(Object.create(proto), gl);
}

// document / window
const byId = new Map();
// ⚠️ 必须把 **index.html 里真实存在的所有 id** 都建出来 ——
// 否则 app.js 在模块级执行 `$('btn-run').onclick = ...` 时会因为 stub 缺元素而抛错，
// 把"探针的缺陷"误报成"产品的缺陷"（实测踩过）。
const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
const HTML_IDS = [...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
const KNOWN_IDS = [...new Set([
  'm-canvas', 'm-overlay', 'm-job', 'm-modes', 'm-list', 'm-status', 'm-hint',
  'm-file', 'm-file-info', 'btn-m-reload', 'btn-m-reset', 'btn-m-wire', 'btn-m-clear',
  'panel-models', 'panel-mesh', 'panel-jobs', 'panel-new', 'panel-doctor',
  'panel-settings', 'health', 'tabs',
  ...HTML_IDS,
])];
KNOWN_IDS.forEach((id) => byId.set(id, new El(id, id === 'm-canvas' ? 'canvas' : 'div')));
// 模拟 HTML 里 select 的默认选中项（取第一个 option 的 value）
const SELS = [...html.matchAll(/<select id="([^"]+)">([\s\S]*?)<\/select>/g)];
SELS.forEach(([, id, body]) => {
  const first = /<option value="([^"]*)"/.exec(body);
  if (first && byId.has(id)) byId.get(id).value = first[1];
});
console.log(`  （stub 建了 ${KNOWN_IDS.length} 个元素，其中来自 index.html 的 ${HTML_IDS.length} 个）`);

globalThis.window = {
  devicePixelRatio: 1, addEventListener: () => {}, removeEventListener: () => {},
};
globalThis.document = {
  getElementById: (id) => byId.get(id) || null,
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: (tag) => new El('', tag),
  addEventListener: () => {},
  body: new El('body', 'body'),
  documentElement: new El('html', 'html'),
  head: new El('head', 'head'),
};
globalThis.self = globalThis.window;
try {
  Object.defineProperty(globalThis, 'navigator', { value: { userAgent: 'node' }, configurable: true });
} catch { /* ignore */ }
globalThis.getComputedStyle = () => ({ width: '800px', height: '600px', display: 'block', visibility: 'visible' });
globalThis.fetch = async () => ({ ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) });
globalThis.Blob = class { constructor() {} };
globalThis.URL = { createObjectURL: () => 'blob:x' };
globalThis.Image = class { set src(_) { if (this.onerror) this.onerror(new Error('no image')); } };
globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };

function loadScript(file) {
  const src = fs.readFileSync(file, 'utf8');
  // 与浏览器一致的全局作用域执行（脚本不是模块）
  new Function('window', 'document', 'fetch', 'Blob', 'URL', 'Image',
               'navigator', 'getComputedStyle', 'requestAnimationFrame',
               'localStorage', 'console', 'setTimeout', 'clearTimeout',
               src)(
    globalThis.window, globalThis.document, globalThis.fetch, globalThis.Blob,
    globalThis.URL, globalThis.Image, globalThis.navigator, globalThis.getComputedStyle,
    globalThis.requestAnimationFrame, globalThis.localStorage, console,
    setTimeout, clearTimeout);
}

let bad = 0;

console.log('=== 加载 viewer.js ===');
try {
  loadScript(path.join(STATIC, 'viewer.js'));
  console.log('  OK; window.GsViewer = ' + (globalThis.window.GsViewer ? 'ok' : '**缺失**'));
  if (!globalThis.window.GsViewer) bad++;
} catch (e) {
  console.log('  **抛出**: ' + e.message + '\n' + e.stack);
  bad++;
}

console.log('\n=== 加载 app.js ===');
try {
  loadScript(path.join(STATIC, 'app.js'));
  console.log('  OK');
} catch (e) {
  console.log('  **抛出**: ' + e.message + '\n' + e.stack);
  bad++;
}

console.log('\n=== 调用 initModelsPage()（模型查看页初始化）===');
try {
  // app.js 是脚本作用域，initModelsPage 可能不在 globalThis 上；
  // 这里通过模拟"点击模型查看页签"来触发它
  const tabs = byId.get('tabs');
  if (typeof globalThis.window.initModelsPage === 'function') {
    await globalThis.window.initModelsPage();
    console.log('  直接调用 initModelsPage() OK');
  } else {
    console.log('  （initModelsPage 不在全局，尝试走页签点击路径）');
  }
} catch (e) {
  console.log('  **抛出**: ' + e.message + '\n' + e.stack);
  bad++;
}

console.log('\n=== 直接调用 GsViewer.init(canvas, cb) ===');
try {
  const canvas = byId.get('m-canvas');
  let statusCalls = 0;
  globalThis.window.GsViewer.init(canvas, () => { statusCalls++; });
  console.log('  init() OK; 状态回调触发次数 = ' + statusCalls);
  const d = globalThis.window.GsViewer.debug();
  console.log('  debug().gl = ' + JSON.stringify(d.gl));
  console.log('  debug().canvas = ' + JSON.stringify(d.canvas));
} catch (e) {
  console.log('  **init() 抛出**: ' + e.message + '\n' + e.stack);
  bad++;
}

console.log('\n结论: ' + (bad ? `**有 ${bad} 处失败**` : '初始化链路全部通过'));
process.exitCode = bad ? 1 : 0;
