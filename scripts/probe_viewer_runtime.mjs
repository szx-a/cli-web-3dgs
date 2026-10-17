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

// 真实 WebGL1 常量号（必须与浏览器一致，否则"常量合法性"和
// "每次调用后检查错误"都无法模拟 —— 曾因用假常量而漏掉 INVALID_ENUM）
const GLV = {
  DEPTH_TEST: 0x0B71, COLOR_BUFFER_BIT: 0x4000, DEPTH_BUFFER_BIT: 0x0100,
  VERTEX_SHADER: 0x8B31, FRAGMENT_SHADER: 0x8B30,
  COMPILE_STATUS: 0x8B81, LINK_STATUS: 0x8B82, VALIDATE_STATUS: 0x8B83,
  ARRAY_BUFFER: 0x8892, ELEMENT_ARRAY_BUFFER: 0x8893, STATIC_DRAW: 0x88E4,
  FLOAT: 0x1406, UNSIGNED_INT: 0x1405, UNSIGNED_BYTE: 0x1401,
  UNSIGNED_SHORT: 0x1403, RGBA: 0x1908, TEXTURE_2D: 0x0DE1,
  TEXTURE_WRAP_S: 0x2802, TEXTURE_WRAP_T: 0x2803, REPEAT: 0x2901,
  TEXTURE_MIN_FILTER: 0x2801, TEXTURE_MAG_FILTER: 0x2800,
  LINEAR_MIPMAP_LINEAR: 0x2703, LINEAR: 0x2601, NEAREST: 0x2600,
  UNPACK_FLIP_Y_WEBGL: 0x9240, POINTS: 0, TRIANGLES: 4, LINES: 1,
  TEXTURE0: 0x84C0, VENDOR: 0x1F00, RENDERER: 0x1F01, VERSION: 0x1F02,
  NO_ERROR: 0, INVALID_ENUM: 0x0500, INVALID_VALUE: 0x0501,
  INVALID_OPERATION: 0x0502, OUT_OF_MEMORY: 0x0505,
  CONTEXT_LOST_WEBGL: 0x9242, ACTIVE_TEXTURE: 0x84E0, NO_ERROR_2: 0,
  DEPTH_FUNC: 0x0B74, LESS: 0x0201, CURRENT_PROGRAM: 0x8B8D,
  MAX_TEXTURE_SIZE: 0x0D33, SAMPLE_BUFFERS: 0x80A8, SAMPLES: 0x80A9,
};

function mkGlStub() {
  let obj = 0;
  // 可用 `--no-uint-index` 模拟"浏览器不提供 OES_element_index_uint"的情形，
  // 用来验证查看器会给出明确错误而不是静默黑屏。
  const glHasUintIndex = !process.argv.includes('--no-uint-index');
  // 错误状态：模拟浏览器"某次调用设置错误码，直到 getError() 读取才清除"
  let errState = 0;
  let quietErr = false;
  const setErr = (code, why) => {
    if (!errState) errState = code;
    if (!quietErr) {
      const names = { 0x0500: 'INVALID_ENUM', 0x0501: 'INVALID_VALUE',
                      0x0502: 'INVALID_OPERATION', 0x0505: 'OUT_OF_MEMORY' };
      console.log(`    ** GL ${names[code] || code} 由 ${why} 触发 **`);
    }
  };
  const G = {
    ...GLV,
    createShader: () => ({ id: 'sh' }),
    shaderSource: () => {}, compileShader: () => {},
    getShaderParameter: () => true, getShaderInfoLog: () => '',
    createProgram: () => PROG, attachShader: () => {}, linkProgram: () => {},
    // ⚠️ getProgramParameter 只接受这几个 pname；传别的必须报 INVALID_ENUM。
    // 这正是浏览器报的那个错 —— mock 必须同样严格，否则测不出来。
    getProgramParameter: (p, pname) => {
      const OKP = [GLV.LINK_STATUS, GLV.VALIDATE_STATUS, 0x8B84 /*ATTACHED_SHADERS*/,
                   0x8B86 /*ACTIVE_UNIFORMS*/, 0x8B89 /*ACTIVE_ATTRIBUTES*/,
                   0x8B8D /*CURRENT_PROGRAM*/, 0x8B80 /*DELETE_STATUS*/];
      if (!OKP.includes(pname)) {
        setErr(0x0500, 'getProgramParameter(pname=' + pname + ')');
        return null;
      }
      return true;
    },
    useProgram: () => {},
    // 故意让"不存在的 uniform"返回 null，模拟真实驱动行为
    getUniformLocation: (p, n) => {
      // 网格着色器的 uniform（已知）+ **泼溅着色器**的 uniform。
      // 泼溅程序用 uView/uProj/uFocal/uViewport，必须也在白名单里，
      // 否则探针会把它们当成"返回 null 的 uniform"误报（它们其实合法）。
      const known = ['uMVP','uPointSize','uUseCol','uUseTex','uRound','uGray','uLight','uTex','uUseNrm',
                     'uView','uProj','uFocal','uViewport'];
      const loc = known.includes(n) ? { n } : null;
      if (!loc) uniformNulls.push(n);
      return loc;
    },
    getAttribLocation: (p, n) => {
      const map = { aPos: 0, aCol: 1, aUV: 2, aNrm: 3,
                    aCorner: 4, aCenter: 5, aCovA: 6, aCovB: 7, aColor: 8 };
      return n in map ? map[n] : -1;
    },
    createBuffer: () => ({ id: 'buf' + (++obj) }),
    // 泼溅渲染器每帧会更新颜色/索引（SH 着色 + 深度排序）——原 stub 缺这一个，
    // 缺它就报 "gl.bufferSubData is not a function"。
    // ⚠️ 只补**真正缺失**的方法：重复定义会覆盖后续带追踪的版本，
    //    导致 gl.__traced 丢失、探针误报"错误溯源未安装"（我踩过）。
    bufferSubData: (target, off, data) =>
      calls.push(['bufferSubData', target, data && data.length]),
    bindBuffer: (t, b) => calls.push(['bindBuffer', t, b && b.id]),
    bufferData: (t, d) => calls.push(['bufferData', t, d && d.byteLength]),
    enableVertexAttribArray: (i) => {
      calls.push(['enableAttrib', i]);
      // 真实 WebGL：索引越界报 INVALID_VALUE；这里也检查一次
      if (i < 0 || i > 15) setErr(0x0501, 'enableVertexAttribArray(' + i + ')');
    },
    disableVertexAttribArray: (i) => {
      calls.push(['disableAttrib', i]);
      if (i < 0 || i > 15) setErr(0x0501, 'disableVertexAttribArray(' + i + ')');
    },
    vertexAttribPointer: (i, n, ty, norm, st, off) => calls.push(['attribPointer', i, n, ty, norm, st, off]),
    vertexAttrib3f: (i, a, b, c) => calls.push(['vertexAttrib3f', i, a, b, c]),
    vertexAttrib2f: (i, a, b) => calls.push(['vertexAttrib2f', i, a, b]),
    enable: () => {}, disable: () => {}, clearColor: () => {},
    clear: () => calls.push(['clear']),
    viewport: (x, y, w, h) => calls.push(['viewport', x, y, w, h]),
    uniformMatrix4fv: (l, tr, m) => {
      calls.push(['uniformMatrix4fv', l && l.n]);
      // 网格/点云路径用 uMVP；**泼溅路径**用 uView + uProj（两个矩阵）。
      // 都要采样：否则跑 3DGS PLY 时 mvpFinite 为空，探针会以
      // "万向锁采样 0 次" 为由退出码 1（误判）。
      if (l && (l.n === 'uMVP' || l.n === 'uView' || l.n === 'uProj') && m) {
        const vals = Array.from(m);
        const finite = vals.every(Number.isFinite);
        mvpFinite.push(finite);
        if (!quiet) {
          const nonzero = vals.filter((v) => v !== 0).length;
          console.log(`  [${l.n}] 有限=${finite} 非零元素=${nonzero}/16`);
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
      // ⚠️ 两条真实 WebGL1 约束（mock 之前没有，所以漏掉了"全黑"的真凶）：
      //   1) type 只能是 UNSIGNED_BYTE(0x1401) 或 UNSIGNED_SHORT(0x1403)；
      //      用 UNSIGNED_INT(0x1405) 必须启用 OES_element_index_uint，否则 INVALID_ENUM。
      //   2) LINES 模式的索引数必须是偶数，否则 INVALID_OPERATION。
      const UNSIGNED_SHORT = 0x1403, UNSIGNED_BYTE = 0x1401, UNSIGNED_INT = 0x1405;
      if (ty !== UNSIGNED_SHORT && ty !== UNSIGNED_BYTE) {
        if (!(ty === UNSIGNED_INT && glHasUintIndex)) {
          setErr(0x0500, `drawElements(type=${ty})`);
        }
      }
      if (m === 1 && c % 2 !== 0) {
        setErr(0x0502, 'drawElements(LINES, ' + c + ') 索引数不是偶数');
      }
    },
    getExtension: (name) => {
      if (name === 'OES_element_index_uint' && glHasUintIndex) return { name };
      // 高斯泼溅渲染器需要实例化绘制（WebGL1 走 ANGLE_instanced_arrays）。
      // 不给这个扩展，`splat.js` 会直接抛"不支持"，于是探测不到泼溅路径。
      // 这里补一个最小实现，并把实例化调用记录下来供断言。
      if (name === 'ANGLE_instanced_arrays') {
        return {
          name,
          vertexAttribDivisorANGLE: (loc, d) => calls.push(['vertexAttribDivisor', loc, d]),
          drawElementsInstancedANGLE: (m, c, t, o, n) => {
            calls.push(['drawElementsInstanced', m, c, n]);
            // ⚠️ 真实 WebGL1 的关键约束：**没启用 `OES_element_index_uint` 就传
            // UNSIGNED_INT 索引，会返回 INVALID_OPERATION 且什么也不画**
            // —— 表现是"黑屏 + 控制台没有异常"。这正是本轮黑屏的原因。
            // stub 必须同样报错，否则探针会放过这个 bug（之前就是如此：
            // 探针只给了扩展、从不校验索引类型与扩展是否匹配）。
            if (t === 0x1405 && !glHasUintIndex) {
              setErr(0x0502,
                'drawElementsInstanced 用了 UNSIGNED_INT 索引，但没有启用 '
                + 'OES_element_index_uint（WebGL1 会静默不画 ⇒ 黑屏）');
            }
            if (m === 1 && c % 2 !== 0) {
              setErr(0x0502, '实例化 drawElements(LINES) 索引数不是偶数');
            }
          },
          drawArraysInstancedANGLE: (m, f, c, n) => {
            calls.push(['drawArraysInstanced', m, f, c, n]);
          },
        };
      }
      return null;
    },
    getError: () => { const e = errState; errState = 0; return e; },
    // 泼溅渲染器画完会 readPixels 抽查是否全黑。stub 返回**背景色**像素，
    // 于是"全黑"能被如实检出（pipe 里没有真实合成结果可读）。
    readPixels: (x, y, w, h, fmt, ty, buf) => {
      for (let i = 0; i + 3 < buf.length; i += 4) {
        buf[i] = 5; buf[i + 1] = 8; buf[i + 2] = 11; buf[i + 3] = 255;
      }
      return undefined;
    },
    getParameter: (pname) => {
      // 合法 pname 集合（含字符串类 VENDOR/RENDERER/VERSION）
      const OKP = [GLV.VENDOR, GLV.RENDERER, GLV.VERSION, GLV.MAX_TEXTURE_SIZE,
                   GLV.SAMPLE_BUFFERS, GLV.SAMPLES, GLV.DEPTH_FUNC,
                   GLV.CURRENT_PROGRAM, 0x0BA2 /*VIEWPORT*/, 0x84E0 /*ACTIVE_TEXTURE*/,
                   0x8894, 0x8895, 0x8CA6, 0x8CA7, 0x8069, 0x8514,
                   0x0D3A, 0x846D, 0x846E, 0x0D50, 0x0D52, 0x0D53, 0x0D54,
                   0x0D55, 0x0D56, 0x0D57, 0x0C10, 0x0C22, 0x0C23];
      if (!OKP.includes(pname)) {
        setErr(0x0500, 'getParameter(' + pname + ')');
        return null;
      }
      if (pname === GLV.VENDOR) return 'MockVendor';
      if (pname === GLV.RENDERER) return 'MockRenderer';
      if (pname === GLV.VERSION) return 'WebGL 1.0 (mock)';
      if (pname === 0x0D3A) return new Int32Array([16384, 16384]);
      return 4096;
    },
    depthFunc: () => {},
    isContextLost: () => false,
    console: { error: (...a) => console.log('  [console.error]', ...a) },
  };
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
  // ⚠️ 必须**缓存**同一个 GL 对象：原来写成 `() => mkGlStub()`，
  // 每次调用都新建一个 ⇒ 探针在外层补的方法（uniform2f 等）补到了另一个对象上，
  // 查看器手里的还是缺方法的旧对象，泼溅渲染照样报 "gl.uniform2f is not a function"。
  getContext: (() => {
    let cached = null;
    return () => { if (!cached) cached = mkGlStub(); return cached; };
  })(),
});
const windowStub = Object.assign(mkTarget(listeners.window), { devicePixelRatio: 1 });
const documentStub = { getElementById: () => canvas };
// debug() 会读计算样式；node 里没有，补一个最小实现，避免"测不了 debug()"
globalThis.getComputedStyle = () => ({ width: '800px', height: '600px',
                                        display: 'block', visibility: 'visible' });

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
// 先加载 splat.js：viewer.js 在遇到 3DGS PLY / .splat 时会调用
// `window.GsSplatRenderer`，没加载就抛"缺少 splat 渲染器"，探测不到泼溅路径。
try {
  const splatSrc = fs.readFileSync(
    path.join('src', 'gsflow', 'web', 'static', 'splat.js'), 'utf8');
  new Function('window', splatSrc)(windowStub);
  console.log('加载 splat.js: ' + (windowStub.GsSplatRenderer ? 'OK' : '**未挂载**'));
} catch (e) {
  console.log('**加载 splat.js 抛出**: ' + e.message);
  process.exit(1);
}

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
  // ---- 补齐泼溅渲染器需要的 GL 接口（**必须在第一次 load 之前**）----
  // 这些方法原 stub 没有（uniform2f / getUniformLocation 等），缺了会在
  // `splat.js::draw` 里抛 "gl.uniform2f is not a function"。
  // 只在缺失时补，绝不覆盖已有的（带错误追踪的）版本 —— 否则探针会误报
  // "错误溯源未安装"（我第一版写进对象字面量，重复定义就把它盖掉了）。
  {
    const g = canvas.getContext('webgl');
    const extra = ['uniform2f', 'uniform3f', 'getUniformLocation', 'getAttribLocation',
                   'uniformMatrix4fv', 'bufferSubData', 'deleteBuffer', 'deleteTexture',
                   'blendFunc', 'depthMask', 'cullFace', 'depthFunc'];
    const added = [];
    for (const n of extra) {
      if (typeof g[n] !== 'function') { g[n] = () => {}; added.push(n); }
    }
    if (added.length) console.log('  补齐 stub 接口: ' + added.join(', '));
  }
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

// debug() 也要跑一遍 —— 它内部会调用 getParameter/getProgramParameter，
// 如果 pname 非法就会报 INVALID_ENUM（浏览器里正是这么暴露的）。
console.log('\n--- debug() 自检 ---');
try {
  const d = V.debug();
  console.log('  gl = ' + JSON.stringify(d.gl));
  console.log('  prog = ' + JSON.stringify(d.prog));
  console.log('  uniforms = ' + JSON.stringify(d.uniforms));
  console.log('  attribs = ' + JSON.stringify(d.attribs));
  console.log('  canvas = ' + JSON.stringify(d.canvas));
  console.log('  obj = ' + JSON.stringify(d.obj));
} catch (e) {
  console.log('  **debug() 抛出**: ' + e.message);
  process.exitCode = 1;
}

// init() 之后严格自检：GL 包装器是否破坏了方法可用性、是否残留错误码。
// 这能验证"我给 GL 加错误溯源"本身没有把查看器弄坏（Edge 上报 INVALID_ENUM 时的首要怀疑）。
console.log('\n--- init() 后严格自检 ---');
{
  const gl = canvas.getContext('webgl');
  // 泼溅渲染器会用到几个原 stub 没有的接口（uniform2f / getUniformLocation 等）。
  // 用**运行时补齐**而不是写进对象字面量：只在缺失时补，绝不覆盖已有的（带追踪的）版本
  // —— 之前我把它们写进字面量，重复定义覆盖了追踪版，探针就误报"错误溯源未安装"。
  const extra = ['uniform2f', 'uniform3f', 'getUniformLocation', 'getAttribLocation',
                 'uniformMatrix4fv', 'bufferSubData', 'deleteBuffer', 'deleteTexture',
                 'blendFunc', 'depthMask', 'cullFace', 'depthFunc'];
  const added = [];
  for (const n of extra) {
    if (typeof gl[n] !== 'function') {
      gl[n] = () => {};
      added.push(n);
    }
  }
  if (added.length) console.log('  补齐 stub 接口: ' + added.join(', '));
  const must = ['getParameter', 'getError', 'enable', 'clear', 'viewport',
                'drawElements', 'drawArrays', 'vertexAttribPointer',
                'enableVertexAttribArray', 'disableVertexAttribArray',
                'uniformMatrix4fv', 'uniform1i', 'useProgram', 'bindBuffer'];
  const broken = must.filter((n) => typeof gl[n] !== 'function');
  console.log('  GL 方法可用性: ' + (broken.length ? '**缺失 ' + broken.join(',') + ' **' : '全部可用'));
  const e = gl.getError();
  console.log('  init 后残留 GL 错误码 = ' + e + (e ? '  **有未清除的错误**' : '  （干净）'));
  if (gl.__traced) {
    console.log('  错误溯源已安装: true');
    // 包装后仍必须能正确取值
    const v = gl.getParameter(gl.VENDOR);
    console.log('  包装后 getParameter(VENDOR) = ' + v);
    const e2 = gl.getError();
    if (e2) { console.log('  **包装后产生错误码 ' + e2 + '**'); process.exitCode = 1; }
  } else {
    console.log('  **错误溯源未安装**');
  }
  if (broken.length || e) process.exitCode = 1;
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
