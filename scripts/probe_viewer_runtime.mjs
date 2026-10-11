/* 在 Node 里加载**真实的 viewer.js**，stub 掉 WebGL/DOM，跑一遍初始化 + 载入模型。
 *
 * 目的：浏览器只显示"黑"，看不到异常。这里把所有 WebGL 调用都记录下来，
 * 于是能回答：
 *   - 初始化是否抛异常？
 *   - 载入模型时调用序列是什么？有没有 uniform/attribute 位置为 null（会静默失效）？
 *   - 渲染时 viewport 尺寸是多少？（0x0 就会黑屏）
 *   - drawArrays/drawElements 的参数对不对？
 *
 * 用法：node scripts/probe_viewer_runtime.mjs <file.glb>
 */
import fs from 'node:fs';
import path from 'node:path';

const glb = process.argv[2];
if (!glb) { console.error('用法: node scripts/probe_viewer_runtime.mjs <file.glb>'); process.exit(2); }

const calls = [];
let uniformNulls = [];
const PROG = { id: 'prog' };

function mkGlStub() {
  let obj = 0;
  const G = {
    // 常量
    DEPTH_TEST: 2929, COLOR_BUFFER_BIT: 16384, DEPTH_BUFFER_BIT: 256,
    VERTEX_SHADER: 35633, FRAGMENT_SHADER: 35632,
    COMPILE_STATUS: 35713, LINK_STATUS: 35714,
    ARRAY_BUFFER: 34962, ELEMENT_ARRAY_BUFFER: 34963, STATIC_DRAW: 35044,
    FLOAT: 5126, UNSIGNED_INT: 5125, UNSIGNED_BYTE: 5121, RGBA: 6408, TEXTURE_2D: 3553,
    TEXTURE_WRAP_S: 10242, TEXTURE_WRAP_T: 10243, REPEAT: 10497,
    TEXTURE_MIN_FILTER: 10241, TEXTURE_MAG_FILTER: 10240,
    LINEAR_MIPMAP_LINEAR: 9987, LINEAR: 9729, UNPACK_FLIP_Y_WEBGL: 37440,
    POINTS: 0, TRIANGLES: 4, LINES: 1,
    // 方法
    createShader: () => ({ id: 'sh' }),
    shaderSource: () => {}, compileShader: () => {},
    getShaderParameter: () => true, getShaderInfoLog: () => '',
    createProgram: () => PROG, attachShader: () => {}, linkProgram: () => {},
    getProgramParameter: () => true, getProgramInfoLog: () => '',
    useProgram: () => {},
    // 故意让"不存在的 uniform"返回 null，模拟真实驱动行为
    getUniformLocation: (p, n) => {
      const known = ['uMVP','uPointSize','uUseCol','uUseTex','uRound','uGray','uLight','uTex','uUseNrm'];
      const loc = known.includes(n) ? { n } : null;
      if (!loc) uniformNulls.push(n);
      return loc;
    },
    getAttribLocation: (p, n) => {
      const map = { aPos: 0, aCol: 1, aUV: 2, aNrm: 3 };
      return n in map ? map[n] : -1;
    },
    createBuffer: () => ({ id: 'buf' + (++obj) }),
    bindBuffer: (t, b) => calls.push(['bindBuffer', t, b && b.id]),
    bufferData: (t, d) => calls.push(['bufferData', t, d && d.byteLength]),
    enableVertexAttribArray: (i) => calls.push(['enableAttrib', i]),
    disableVertexAttribArray: (i) => calls.push(['disableAttrib', i]),
    vertexAttribPointer: (i, n, ty, norm, st, off) => calls.push(['attribPointer', i, n, ty, norm, st, off]),
    vertexAttrib3f: (i, a, b, c) => calls.push(['vertexAttrib3f', i, a, b, c]),
    vertexAttrib2f: (i, a, b) => calls.push(['vertexAttrib2f', i, a, b]),
    enable: () => {}, disable: () => {}, clearColor: () => {},
    clear: () => calls.push(['clear']),
    viewport: (x, y, w, h) => calls.push(['viewport', x, y, w, h]),
    uniformMatrix4fv: (l, tr, m) => {
      calls.push(['uniformMatrix4fv', l && l.n]);
      if (l && l.n === 'uMVP' && m) {
        const vals = Array.from(m);
        const finite = vals.every(Number.isFinite);
        // MVP 是否退化：全 0 或含 NaN 都会导致什么都画不出来
        const nonzero = vals.filter((v) => v !== 0).length;
        console.log('  [uMVP] 有限=' + finite + ' 非零元素=' + nonzero + '/16');
        console.log('        ' + vals.map((v) => v.toFixed(3)).join(', '));
      }
    },
    uniform1f: (l, v) => calls.push(['uniform1f', l && l.n, v]),
    uniform1i: (l, v) => calls.push(['uniform1i', l && l.n, v]),
    uniform3f: (l, a, b, c) => calls.push(['uniform3f', l && l.n, a, b, c]),
    activeTexture: () => {}, bindTexture: () => {}, pixelStorei: () => {},
    texImage2D: () => {}, texParameteri: () => {}, generateMipmap: () => {},
    createTexture: () => ({ id: 'tex' }), deleteTexture: () => {},
    deleteBuffer: () => {},
    drawArrays: (m, f, c) => calls.push(['drawArrays', m, f, c]),
    drawElements: (m, c, ty, off) => {
      calls.push(['drawElements', m, c, ty, off]);
      // 模拟真实 WebGL 的校验：LINES 模式的索引数必须是偶数，
      // 否则报 INVALID_OPERATION 并且整帧不画（这正是线框全黑的成因）
      if (m === 1 && c % 2 !== 0) {
        console.log('  ** INVALID_OPERATION: LINES 索引数 ' + c + ' 不是偶数 -> 整帧不绘制 **');
        lastError = 1282;
      }
    },
    getError: () => { const e = lastError; lastError = 0; return e; },
    console: { error: (...a) => console.log('  [console.error]', ...a) },
  };
  let lastError = 0;
  return G;
}

// ---- DOM / window stubs ----
const canvas = {
  clientWidth: 800, clientHeight: 600, width: 800, height: 600,
  getContext: () => mkGlStub(),
  addEventListener: () => {},
};
const windowStub = {
  devicePixelRatio: 1,
  addEventListener: () => {},
};
const documentStub = { getElementById: () => canvas };

globalThis.window = windowStub;
globalThis.document = documentStub;
globalThis.self = windowStub;
// node 24 里 navigator 是只读 getter，直接赋值会抛 TypeError
try {
  Object.defineProperty(globalThis, 'navigator', {
    value: { userAgent: 'node' }, configurable: true, writable: true,
  });
} catch { /* 已有就用现成的 */ }

// fetch：把本地文件当作 HTTP 响应喂进去
const buf = fs.readFileSync(glb);
globalThis.fetch = async () => ({
  ok: true, status: 200,
  arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
});
globalThis.Blob = class { constructor() {} };
globalThis.URL = { createObjectURL: () => 'blob:stub' };
globalThis.Image = class { set src(_) { if (this.onerror) this.onerror(new Error('no image')); } };

// ---- 加载真实 viewer.js ----
const src = fs.readFileSync(path.join('src', 'gsflow', 'web', 'static', 'viewer.js'), 'utf8');
const mod = { exports: {} };
try {
  // viewer.js 是脚本（不是模块），末尾给 window.GsViewer 赋值
  new Function('window', 'document', 'fetch', 'Blob', 'URL', 'Image', 'navigator', src)(
    windowStub, documentStub, globalThis.fetch, globalThis.Blob, globalThis.URL,
    globalThis.Image, globalThis.navigator);
  console.log('加载 viewer.js: OK');
} catch (e) {
  console.log('**加载 viewer.js 抛出**: ' + e.message);
  process.exit(1);
}

const V = windowStub.GsViewer;
if (!V) { console.log('**window.GsViewer 未定义**'); process.exit(1); }
console.log('window.GsViewer 方法: ' + Object.keys(V).join(', '));

let statusLog = [];
try {
  V.init(canvas, (s) => { statusLog.push(s); console.log('  [status] ' + s); });
  console.log('init() OK');
} catch (e) {
  console.log('**init() 抛出**: ' + e.message + '\n' + e.stack);
  process.exit(1);
}

try {
  const info = await V.load('/api/artifacts/x/' + path.basename(glb), 'glb');
  console.log('load() 返回: ' + JSON.stringify(info));
} catch (e) {
  console.log('**load() 抛出**: ' + e.message + '\n' + e.stack);
}

// 线框回归：原先 drawElements(gl.LINES, 三角形索引数) 会因为
// 索引数不是偶数而报 INVALID_OPERATION，**整帧不画**（线框全黑）。
console.log('\n--- 切换线框 ---');
calls.length = 0;
try {
  const on = V.toggleWire();
  console.log('toggleWire() -> ' + on);
} catch (e) {
  console.log('**toggleWire() 抛出**: ' + e.message);
}
try {
  V.toggleWire();   // 切回实体
} catch (e) {
  console.log('**toggleWire() 第二次抛出**: ' + e.message);
}

console.log('\n--- 渲染调用序列（关键项）---');
for (const c of calls) {
  if (['viewport','drawArrays','drawElements','uniform1i','uniform3f','enableAttrib','attribPointer','clear'].includes(c[0])) {
    console.log('  ' + JSON.stringify(c));
  }
}
if (uniformNulls.length) {
  console.log('\n**getUniformLocation 返回 null 的 uniform**（这些 uniform 会被静默忽略）: '
              + [...new Set(uniformNulls)].join(', '));
}
