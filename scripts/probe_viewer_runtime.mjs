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
let mvpFinite = [];
let quiet = false;      // 大量拖拽采样时关掉逐条打印
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
        mvpFinite.push(finite);
        if (!quiet) {
          const nonzero = vals.filter((v) => v !== 0).length;
          console.log('  [uMVP] 有限=' + finite + ' 非零元素=' + nonzero + '/16');
        }
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
// 事件监听要真的记下来，才能模拟拖拽（万向锁回归依赖它）
const listeners = { canvas: {}, window: {} };
function mkTarget(store) {
  return {
    addEventListener: (t, fn) => { (store[t] = store[t] || []).push(fn); },
    removeEventListener: () => {},
    setPointerCapture: () => {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
  };
}
const canvas = Object.assign(mkTarget(listeners.canvas), {
  clientWidth: 800, clientHeight: 600, width: 800, height: 600,
  getContext: () => mkGlStub(),
});
const windowStub = Object.assign(mkTarget(listeners.window), { devicePixelRatio: 1 });
const documentStub = { getElementById: () => canvas };

/** 模拟一次鼠标拖拽（按下 -> 移动 -> 抬起） */
function draggingSim(target, dx, dy) {
  const fire = (store, type, ev) => (store[type] || []).forEach((f) => f(ev));
  fire(listeners.canvas, 'mousedown', { button: 0, shiftKey: false, clientX: 400, clientY: 300, preventDefault() {} });
  fire(listeners.window, 'mousemove', { clientX: 400 + dx, clientY: 300 + dy });
  fire(listeners.window, 'mouseup', {});
}

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

// kind 必须按扩展名判断 —— 写死 'glb' 会让 .ply 走 GLB 分支，
// 报出误导性的"不是 GLB 文件"（实测踩过）。
const kindOf = (p) => (p.toLowerCase().endsWith('.ply') ? 'ply' : 'glb');
try {
  const info = await V.load('/api/artifacts/x/' + path.basename(glb), kindOf(glb));
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

// 万向锁回归：拖到 pitch = ±90°（相机方向与 up 共线）时，
// 原实现 `norm(cross(up, z))` 得到零向量 -> 视图矩阵含 NaN -> 画面消失。
// 这里模拟"用户一直往下拖"直到越界，确认矩阵始终有限。
console.log('\n--- 万向锁回归（拖过顶/底）---');
calls.length = 0;
mvpFinite = [];
quiet = true;
// 模拟连续拖拽：每次向下拖 200px（0.008 rad/px -> 1.6 rad，足以越过 ±π/2）
try {
  for (let i = 0; i < 6; i++) {
    draggingSim(canvas, 0, 200);
  }
  for (let i = 0; i < 12; i++) {
    draggingSim(canvas, 0, -200);
  }
} catch (e) {
  console.log('**拖拽模拟抛出**: ' + e.message);
}
quiet = false;
const bad = mvpFinite.filter((x) => !x).length;
console.log(`  采样 ${mvpFinite.length} 次 uMVP，含 NaN/Inf 的次数 = ${bad}`
            + (bad ? '  **仍有退化**' : '  （全部有限，万向锁已修）'));

// 本地文件路径回归：GUI 的"直接打开本地 .ply/.glb"走的是 loadBuffer()，
// 与远端 load() 共用同一条解析逻辑。这里直接以"内存 buffer"方式载入同一个文件，
// 确认本地路径也能解析（否则用户选了文件仍然黑屏）。
console.log('\n--- 本地文件路径 loadBuffer() ---');
try {
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const r2 = await V.loadBuffer(ab, glb.split(/[\\/]/).pop(), kindOf(glb));
  console.log('  loadBuffer() 返回: ' + JSON.stringify(r2));
  if (!r2 || !r2.vertices) {
    console.log('  **本地文件路径解析失败**');
    process.exitCode = 1;
  }
} catch (e) {
  console.log('  **loadBuffer() 抛出**: ' + e.message);
  process.exitCode = 1;
}

// clear() 必须能清空（"清空视图"按钮用）
try {
  V.clear();
  console.log('  clear() OK');
} catch (e) {
  console.log('  **clear() 抛出**: ' + e.message);
  process.exitCode = 1;
}

// 退出码：供 pytest 做回归（非 0 表示有退化）
process.exitCode = (bad > 0 || mvpFinite.length === 0) ? 1 : (process.exitCode || 0);

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
