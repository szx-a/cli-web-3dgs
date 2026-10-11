/* gsflow 模型查看器 —— 纯 WebGL，无 CDN、无构建步骤，离线可用。
 *
 * 为什么不用 three.js：本项目的 GUI 一直坚持"离线可用"（viewer.html 是单文件产物，
 * 双击就能打开）。引入 CDN 会让 GUI 在断网时直接白屏，违反这条既有约束。
 * 而我们要看的两种东西都很简单：
 *   * glTF 2.0 二进制（TRIANGLES / POINTS 图元 + 顶点色 + 可选贴图）—— 自己解析
 *   * PLY 点云（binary_little_endian）—— 自己解析
 * 所以直接写最小 WebGL 渲染器更稳。
 *
 * 三种模式：
 *   points   —— 点云（model.glb 或 model.ply）
 *   mesh     —— 三角网格 + 顶点色（model_mesh.glb）
 *   textured —— 带贴图网格（model_mesh_textured.glb）
 */
'use strict';

/* ---------------------------------------------------------------------------
 * ⚠️ 支持的容器格式（**界面层必须先过滤**，否则会"打不开且只显示黑"）：
 *     .glb   glTF 2.0 二进制（TRIANGLES / POINTS + COLOR_0 + NORMAL + TEXCOORD_0）
 *     .ply   binary_little_endian 点云（x/y/z + red/green/blue 或 f_dc_*）
 *   **不支持 .obj / .spz** —— 虽然我们也会导出它们（给 Blender 等外部工具用），
 *   但要把它们从"可查看"列表里排除，不能让用户点了才发现打不开。
 * ------------------------------------------------------------------------- */

/* ============================ 数学（列主序 4x4） ============================ */
function m4identity() { return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]); }

function m4mul(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
      o[i * 4 + j] = s;
    }
  }
  return o;
}

function m4perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function m4lookAt(eye, center, up) {
  const z = norm(sub(eye, center));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
function norm(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/l, a[1]/l, a[2]/l]; }

/* ============================ glTF 2.0 解析 ============================ */
const GLTF_COMP = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array,
                    5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const GLTF_ITEMS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

function parseGlb(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('不是 GLB 文件');
  const total = dv.getUint32(8, true);
  let off = 12, json = null, bin = null;
  while (off + 8 <= Math.min(total, buf.byteLength)) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    const start = off + 8;
    if (type === 0x4E4F534A) {
      json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, start, len)));
    } else if (type === 0x004E4942) {
      bin = new Uint8Array(buf, start, len);
    }
    off = start + len;
  }
  if (!json) throw new Error('GLB 缺少 JSON chunk');
  return { gltf: json, bin };
}

function readAccessor(gltf, bin, idx) {
  const acc = gltf.accessors[idx];
  const n = GLTF_ITEMS[acc.type];
  const Ctor = GLTF_COMP[acc.componentType];
  if (!Ctor) throw new Error('不支持的 componentType ' + acc.componentType);
  const bv = gltf.bufferViews[acc.bufferView];
  const base = (bv.byteOffset || 0) + (acc.byteOffset || 0);
  const count = acc.count * n;
  const stride = bv.byteStride || 0;
  let out;
  if (!stride || stride === n * Ctor.BYTES_PER_ELEMENT) {
    // 紧密排列：考虑 bin 的字节偏移对齐
    out = new Ctor(bin.buffer, bin.byteOffset + base, count);
  } else {
    out = new Ctor(count);
    const bpe = Ctor.BYTES_PER_ELEMENT;
    for (let i = 0; i < acc.count; i++) {
      const src = new Ctor(bin.buffer, bin.byteOffset + base + i * stride, n);
      out.set(src, i * n);
    }
  }
  return { data: out, n, count: acc.count, normalized: !!acc.normalized,
           componentType: acc.componentType };
}

/** 把 glTF 里的一种属性取成 Float32Array（顶点数 × n），并按需反归一化。 */
function accessorToFloat(gltf, bin, idx) {
  const a = readAccessor(gltf, bin, idx);
  const out = new Float32Array(a.count * a.n);
  if (a.data instanceof Float32Array) {
    out.set(a.data.subarray(0, out.length));
  } else {
    let scale = 1;
    if (a.normalized) {
      if (a.componentType === 5121) scale = 1 / 255;
      else if (a.componentType === 5123) scale = 1 / 65535;
      else if (a.componentType === 5120) scale = 1 / 127;
      else if (a.componentType === 5122) scale = 1 / 32767;
    }
    for (let i = 0; i < out.length; i++) out[i] = a.data[i] * scale;
  }
  return out;
}

/** 从 GLB 里取一个可渲染的网格：{mode, position, color, uv, indices, textureUrl} */
function glbToRenderable(buf, textureResolver) {
  const { gltf, bin } = parseGlb(buf);
  const prim = gltf.meshes[0].primitives[0];
  const attrs = prim.attributes || {};
  const position = accessorToFloat(gltf, bin, attrs.POSITION);
  let color = attrs.COLOR_0 !== undefined ? accessorToFloat(gltf, bin, attrs.COLOR_0) : null;
  if (color && color.length / 3 !== position.length / 3) color = null;
  const uv = attrs.TEXCOORD_0 !== undefined ? accessorToFloat(gltf, bin, attrs.TEXCOORD_0) : null;
  // NORMAL：有就用（平滑着色）；没有则留空，着色器退回一个固定方向
  let normal = attrs.NORMAL !== undefined ? accessorToFloat(gltf, bin, attrs.NORMAL) : null;
  if (normal && normal.length / 3 !== position.length / 3) normal = null;
  let indices = null;
  if (prim.indices !== undefined) {
    const a = readAccessor(gltf, bin, prim.indices);
    indices = a.data instanceof Uint32Array ? a.data
            : a.data instanceof Uint16Array ? new Uint32Array(a.data)
            : Uint32Array.from(a.data);
  }
  // 贴图：glTF 的 images[].bufferView 指向内嵌 PNG
  let textureUrl = null;
  const mat = (gltf.materials || [])[0];
  const hasTex = !!(mat && mat.pbrMetallicRoughness && mat.pbrMetallicRoughness.baseColorTexture);
  if (hasTex && gltf.images && gltf.images.length) {
    const img = gltf.images[0];
    if (img.bufferView !== undefined) {
      const bv = gltf.bufferViews[img.bufferView];
      const bytes = bin.subarray(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength);
      textureUrl = URL.createObjectURL(new Blob([bytes], { type: img.mimeType || 'image/png' }));
    } else if (img.uri) {
      textureUrl = textureResolver ? textureResolver(img.uri) : img.uri;
    }
  }
  return { mode: prim.mode === undefined ? 4 : prim.mode, position, color, uv, normal,
           indices, textureUrl };
}

/* ============================ PLY 解析（点云） ============================ */
/** 解析 binary_little_endian PLY 的 x/y/z + 颜色（red/green/blue 或 f_dc_*）。 */
function parsePly(buf) {
  const head = new Uint8Array(buf, 0, Math.min(buf.byteLength, 65536));
  // 头是 ASCII，逐行找 end_header
  let headEnd = -1, headText = '';
  const td = new TextDecoder();
  for (let i = 0; i + 10 <= head.length; i++) {
    if (head[i] === 0x65 && td.decode(head.subarray(i, i + 10)) === 'end_header') {
      // 跳过该行剩余（\n 或 \r\n）
      let j = i + 10;
      while (j < head.length && head[j] !== 0x0A) j++;
      headEnd = j + 1;
      headText = td.decode(head.subarray(0, headEnd));
      break;
    }
  }
  if (headEnd < 0) throw new Error('PLY 头里找不到 end_header');
  const lines = headText.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const fmt = (lines.find((l) => l.startsWith('format')) || '').split(/\s+/)[1] || '';
  if (!fmt.startsWith('binary')) throw new Error('只支持 binary PLY（当前 ' + fmt + '）');
  const le = fmt.includes('little');
  const vcount = parseInt((lines.find((l) => l.startsWith('element vertex')) || '').split(/\s+/)[2], 10);
  if (!vcount) throw new Error('PLY 没有 element vertex');

  // 属性列表（只取 vertex 元素）
  const props = [];
  let inVertex = false;
  for (const l of lines) {
    if (l.startsWith('element vertex')) { inVertex = true; continue; }
    if (l.startsWith('element ') && !l.startsWith('element vertex')) { inVertex = false; continue; }
    if (inVertex && l.startsWith('property ')) {
      const p = l.split(/\s+/);
      props.push({ type: p[1], name: p[2] });
    }
  }
  const SIZE = { char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2,
                 int16: 2, uint16: 2, int: 4, uint: 4, int32: 4, uint32: 4,
                 float: 4, float32: 4, double: 8, float64: 8 };
  let stride = 0;
  const offs = {};
  for (const p of props) { offs[p.name] = stride; stride += SIZE[p.type] || 4; }
  const need = ['x', 'y', 'z'];
  for (const k of need) if (offs[k] === undefined) throw new Error('PLY 缺少 ' + k);

  const dv = new DataView(buf, headEnd);
  const get = (name) => {
    const p = props.find((q) => q.name === name);
    if (!p) return null;
    const t = p.type;
    const o = offs[name];
    const read = (i) => {
      const b = i * stride + o;
      switch (t) {
        case 'float': case 'float32': return dv.getFloat32(b, le);
        case 'double': case 'float64': return dv.getFloat64(b, le);
        case 'uchar': case 'uint8': return dv.getUint8(b);
        case 'char': case 'int8': return dv.getInt8(b);
        case 'ushort': case 'uint16': return dv.getUint16(b, le);
        case 'short': case 'int16': return dv.getInt16(b, le);
        case 'uint': case 'uint32': return dv.getUint32(b, le);
        default: return dv.getInt32(b, le);
      }
    };
    return read;
  };
  const rx = get('x'), ry = get('y'), rz = get('z');
  const position = new Float32Array(vcount * 3);
  for (let i = 0; i < vcount; i++) {
    position[i*3] = rx(i); position[i*3+1] = ry(i); position[i*3+2] = rz(i);
  }
  // 颜色：优先 red/green/blue（0~255），否则 f_dc_*（SH 直流项）
  let color = null;
  const rr = get('red'), gg = get('green'), bb = get('blue');
  if (rr && gg && bb) {
    color = new Float32Array(vcount * 3);
    for (let i = 0; i < vcount; i++) {
      color[i*3] = rr(i) / 255; color[i*3+1] = gg(i) / 255; color[i*3+2] = bb(i) / 255;
    }
  } else {
    const f0 = get('f_dc_0'), f1 = get('f_dc_1'), f2 = get('f_dc_2');
    if (f0 && f1 && f2) {
      const C0 = 0.28209479177387814;
      color = new Float32Array(vcount * 3);
      for (let i = 0; i < vcount; i++) {
        color[i*3] = Math.min(1, Math.max(0, 0.5 + C0 * f0(i)));
        color[i*3+1] = Math.min(1, Math.max(0, 0.5 + C0 * f1(i)));
        color[i*3+2] = Math.min(1, Math.max(0, 0.5 + C0 * f2(i)));
      }
    }
  }
  return { mode: 0, position, color, uv: null, indices: null, textureUrl: null };
}

/* ============================ WebGL 渲染器 ============================ */
const VS = `
attribute vec3 aPos;
attribute vec3 aCol;
attribute vec2 aUV;
attribute vec3 aNrm;
uniform mat4 uMVP;
uniform float uPointSize;
uniform bool uUseCol;
uniform bool uUseNrm;
varying vec3 vCol;
varying vec2 vUV;
varying vec3 vNrm;
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  gl_PointSize = uPointSize;
  vCol = uUseCol ? aCol : vec3(0.82, 0.84, 0.88);
  vUV = aUV;
  vNrm = uUseNrm ? aNrm : vec3(0.0, 0.0, 1.0);
}`;

const FS = `
precision mediump float;
uniform bool uUseTex;
uniform bool uUseCol;
uniform bool uUseNrm;
uniform bool uRound;      // 点云用圆点
uniform bool uGray;       // 网格无顶点色时用灰色
uniform vec3 uLight;
uniform sampler2D uTex;
varying vec3 vCol;
varying vec2 vUV;
varying vec3 vNrm;
void main() {
  if (uRound) {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
  }
  vec3 base = uGray ? vec3(0.72, 0.74, 0.78) : vCol;
  if (uUseTex) {
    vec4 t = texture2D(uTex, vUV);
    // PNG 是 sRGB 编码，而顶点色按线性处理 —— 不做 sRGB->线性会显得发灰发白
    base = pow(t.rgb, vec3(2.2));
  }
  // ⚠️ 光照必须用**逐像素的法线**。原来写的是
  //     lam = 0.55 + 0.45 * max(dot(normalize(uLight), vec3(0,0,1)), 0)
  //   两个方向都与像素无关 -> 全屏同一个常数 -> 等于**没有光照**，
  //   网格看上去就是一片死平的色块（用户反馈"白/杂乱"的观感来源之一）。
  // 现在用顶点法线（半兰伯特 + 环境项），并叠一点点边缘增强让面结构可辨。
  vec3 n = normalize(vNrm);
  float diff = max(dot(n, normalize(uLight)), 0.0);
  float lam = 0.30 + 0.70 * diff;
  vec3 lin = base * lam;
  // 线性 -> sRGB，否会整体偏亮、发白
  gl_FragColor = vec4(pow(clamp(lin, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}`;

function createRenderer(canvas) {
  const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
  if (!gl) throw new Error('浏览器不支持 WebGL');

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error('着色器编译失败：' + gl.getShaderInfoLog(s));
    }
    return s;
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error('着色器链接失败：' + gl.getProgramInfoLog(prog));
  }
  gl.useProgram(prog);
  const U = {};
  ['uMVP', 'uPointSize', 'uUseCol', 'uUseTex', 'uRound', 'uGray', 'uLight', 'uTex',
   'uUseNrm'].forEach((n) => { U[n] = gl.getUniformLocation(prog, n); });
  const A = {
    pos: gl.getAttribLocation(prog, 'aPos'),
    col: gl.getAttribLocation(prog, 'aCol'),
    uv: gl.getAttribLocation(prog, 'aUV'),
    nrm: gl.getAttribLocation(prog, 'aNrm'),
  };
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.02, 0.03, 0.045, 1);

  return { gl, prog, U, A };
}

/** 一个可渲染对象：缓冲 + 元信息 */
function makeMesh(R, model, wireframe) {
  const { gl } = R;
  const vbuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbuf);
  gl.bufferData(gl.ARRAY_BUFFER, model.position, gl.STATIC_DRAW);
  let cbuf = null;
  if (model.color) {
    cbuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cbuf);
    gl.bufferData(gl.ARRAY_BUFFER, model.color, gl.STATIC_DRAW);
  }
  let ubuf = null;
  if (model.uv) {
    ubuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, ubuf);
    gl.bufferData(gl.ARRAY_BUFFER, model.uv, gl.STATIC_DRAW);
  }
  let nbuf = null;
  if (model.normal) {
    nbuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nbuf);
    gl.bufferData(gl.ARRAY_BUFFER, model.normal, gl.STATIC_DRAW);
  }
  let ibuf = null, indexCount = 0;
  if (model.indices) {
    ibuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, model.indices, gl.STATIC_DRAW);
    indexCount = model.indices.length;
  }
  // 线框用的线段索引：每个三角形贡献 3 条边 (a-b, b-c, c-a)。
  // 必须单独建缓冲 —— 直接拿三角形索引去 drawElements(gl.LINES) 会报
  // INVALID_OPERATION（索引数不是 2 的倍数）并让**整帧不画**。
  let lineIbuf = null, lineIndexCount = 0;
  if (model.indices && model.indices.length >= 3) {
    const tri = model.indices;
    const nTri = Math.floor(tri.length / 3);
    const lines = new Uint32Array(nTri * 6);
    for (let t = 0; t < nTri; t++) {
      const a = tri[t * 3], b = tri[t * 3 + 1], c = tri[t * 3 + 2];
      lines[t * 6] = a; lines[t * 6 + 1] = b;
      lines[t * 6 + 2] = b; lines[t * 6 + 3] = c;
      lines[t * 6 + 4] = c; lines[t * 6 + 5] = a;
    }
    lineIbuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineIbuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, lines, gl.STATIC_DRAW);
    lineIndexCount = lines.length;
  }
  // 包围盒 -> 自动取景
  let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < model.position.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = model.position[i + k];
      if (v < mn[k]) mn[k] = v;
      if (v > mx[k]) mx[k] = v;
    }
  }
  const center = [(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2];
  const radius = Math.max(1e-6, 0.5 * Math.hypot(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]));
  return { vbuf, cbuf, ubuf, nbuf, ibuf, lineIbuf, indexCount, lineIndexCount,
           count: model.position.length / 3, mode: model.mode,
           center, radius, wireframe,
           useCol: !!model.color, useUV: !!model.uv, useNrm: !!model.normal,
           texture: null };
}

function uploadTexture(R, url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const { gl } = R;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.generateMipmap(gl.TEXTURE_2D);
      resolve(tex);
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

/* ============================ 对外接口 ============================ */
window.GsViewer = (function () {
  let R = null, canvas = null, obj = null;
  let yaw = 0.6, pitch = 0.45, dist = 3, target = [0, 0, 0];
  let dragging = 0, lastX = 0, lastY = 0, wire = false;
  let onStatus = null;

  function resize() {
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  function draw() {
    if (!R || !canvas) return;
    resize();
    const { gl } = R;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!obj) return;

    const aspect = canvas.width / Math.max(1, canvas.height);
    const eye = [
      target[0] + dist * Math.cos(pitch) * Math.sin(yaw),
      target[1] + dist * Math.sin(pitch),
      target[2] + dist * Math.cos(pitch) * Math.cos(yaw),
    ];
    // 注意：GLB 已做 y/z 翻转使其正立；相机 up 用 +Y
    const view = m4lookAt(eye, target, [0, 1, 0]);
    const proj = m4perspective(Math.PI / 4, aspect, dist * 1e-3, dist * 100);
    const mvp = m4mul(proj, view);

    gl.useProgram(R.prog);
    gl.uniformMatrix4fv(R.U.uMVP, false, mvp);
    gl.uniform1f(R.U.uPointSize, Math.max(1.5, canvas.height / 420));
    gl.uniform1i(R.U.uUseCol, obj.useCol ? 1 : 0);
    gl.uniform1i(R.U.uUseTex, obj.texture ? 1 : 0);
    gl.uniform1i(R.U.uRound, obj.mode === 0 ? 1 : 0);
    gl.uniform1i(R.U.uGray, (obj.mode !== 0 && !obj.useCol && !obj.texture) ? 1 : 0);
    gl.uniform1i(R.U.uUseNrm, obj.useNrm ? 1 : 0);
    gl.uniform3f(R.U.uLight, 0.4, 0.7, 0.6);

    gl.bindBuffer(gl.ARRAY_BUFFER, obj.vbuf);
    gl.enableVertexAttribArray(R.A.pos);
    gl.vertexAttribPointer(R.A.pos, 3, gl.FLOAT, false, 0, 0);

    if (obj.cbuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, obj.cbuf);
      gl.enableVertexAttribArray(R.A.col);
      gl.vertexAttribPointer(R.A.col, 3, gl.FLOAT, false, 0, 0);
    } else {
      gl.disableVertexAttribArray(R.A.col);
      gl.vertexAttrib3f(R.A.col, 0.82, 0.84, 0.88);
    }
    if (obj.ubuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, obj.ubuf);
      gl.enableVertexAttribArray(R.A.uv);
      gl.vertexAttribPointer(R.A.uv, 2, gl.FLOAT, false, 0, 0);
    } else {
      gl.disableVertexAttribArray(R.A.uv);
      gl.vertexAttrib2f(R.A.uv, 0, 0);
    }
    if (obj.nbuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, obj.nbuf);
      gl.enableVertexAttribArray(R.A.nrm);
      gl.vertexAttribPointer(R.A.nrm, 3, gl.FLOAT, false, 0, 0);
    } else {
      gl.disableVertexAttribArray(R.A.nrm);
      gl.vertexAttrib3f(R.A.nrm, 0, 0, 1);
    }
    if (obj.texture) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, obj.texture);
      gl.uniform1i(R.U.uTex, 0);
    }

    if (obj.mode === 0) {
      gl.drawArrays(gl.POINTS, 0, obj.count);
    } else if (wire && obj.lineIbuf) {
      // ⚠️ 线框**必须用专门的线段索引缓冲**。踩坑：原先直接
      //     drawElements(gl.LINES, obj.indexCount, ...)
      //   把"三角形索引表"当成线段表用 —— 索引数不是 2 的倍数时
      //   WebGL 会报 INVALID_OPERATION 并且**整帧什么都不画**（表现为全黑），
      //   而不是画错。现在改为在载入时预生成 (3*面数) 条边的 LINES 索引。
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, obj.lineIbuf);
      gl.drawElements(gl.LINES, obj.lineIndexCount, gl.UNSIGNED_INT, 0);
    } else if (obj.ibuf) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, obj.ibuf);
      gl.drawElements(gl.TRIANGLES, obj.indexCount, gl.UNSIGNED_INT, 0);
    } else {
      gl.drawArrays(gl.TRIANGLES, 0, obj.count);
    }

    // 把 WebGL 错误暴露出来 —— 否则一切静默失败，界面只剩"黑"，无从排查。
    const err = gl.getError();
    if (err !== 0) {
      const names = { 1280: 'INVALID_ENUM', 1281: 'INVALID_VALUE', 1282: 'INVALID_OPERATION',
                      1285: 'OUT_OF_MEMORY', 1286: 'INVALID_FRAMEBUFFER_OPERATION',
                      37442: 'CONTEXT_LOST_WEBGL' };
      const msg = (names[err] || ('0x' + err.toString(16)));
      if (onStatus) onStatus('⚠ WebGL 错误: ' + msg);
      if (window.console) console.error('[GsViewer] WebGL error', msg);
    }
  }

  function bindEvents() {
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('mousedown', (e) => {
      dragging = (e.button === 2 || e.shiftKey) ? 2 : 1;
      lastX = e.clientX; lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => { dragging = 0; });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (dragging === 1) {
        yaw -= dx * 0.008;
        pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * 0.008));
      } else {
        // 平移：沿相机的右/上方向在屏幕平面内移动目标点
        const s = dist * 0.0016;
        const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
        target[0] -= right[0] * dx * s; target[2] -= right[2] * dx * s;
        target[1] += dy * s;
      }
      draw();
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      dist = Math.max(0.02, dist * (1 + Math.sign(e.deltaY) * 0.12));
      draw();
    }, { passive: false });
    window.addEventListener('resize', () => draw());
  }

  return {
    init(canvasEl, statusCb) {
      canvas = canvasEl;
      onStatus = statusCb;
      R = createRenderer(canvas);
      bindEvents();
      draw();
    },
    reset() {
      if (!obj) return;
      target = obj.center.slice();
      dist = obj.radius * 2.8;
      yaw = 0.6; pitch = 0.45;
      draw();
    },
    toggleWire() { wire = !wire; draw(); return wire; },
    /** 载入一个模型：kind 为 'glb' 或 'ply' */
    async load(url, kind) {
      if (!R) throw new Error('查看器未初始化');
      if (onStatus) onStatus('载入中…');
      const res = await fetch(url);
      if (!res.ok) throw new Error('下载失败 HTTP ' + res.status
                                   + '（文件可能已被删除 —— 重跑一次导出即可）');
      const buf = await res.arrayBuffer();
      if (!buf || buf.byteLength < 64) {
        throw new Error('文件为空或过小（' + (buf ? buf.byteLength : 0)
                        + ' 字节）—— 该产物可能未成功导出');
      }
      let model;
      try {
        const low = url.toLowerCase();
        if (kind === 'glb' || low.endsWith('.glb')) {
          model = glbToRenderable(buf, null);
        } else if (low.endsWith('.ply')) {
          model = parsePly(buf);
        } else {
          // ⚠️ 只支持 glb / ply 两种容器。OBJ / SPZ 等**必须在界面层就过滤掉** ——
          // 否则这里会走进 GLB 解析分支、抛一个含糊的错误，用户只看到"黑"。
          throw new Error('查看器不支持该格式（仅 GLB 与 PLY）：'
                          + url.split('/').pop());
        }
      } catch (e) {
        throw new Error('解析失败：' + e.message);
      }
      if (!model || !model.position || model.position.length < 3) {
        throw new Error('解析后没有任何顶点（文件可能只含空网格）');
      }
      if (obj) {
        const { gl } = R;
        [obj.vbuf, obj.cbuf, obj.ubuf, obj.nbuf, obj.ibuf, obj.lineIbuf]
          .forEach((b) => { if (b) gl.deleteBuffer(b); });
        if (obj.texture) gl.deleteTexture(obj.texture);
      }
      obj = makeMesh(R, model, wire);
      if (model.textureUrl) obj.texture = await uploadTexture(R, model.textureUrl);
      this.reset();
      draw();
      return { vertices: obj.count, triangles: obj.indexCount / 3,
               mode: obj.mode, textured: !!obj.texture, colors: obj.useCol,
               normals: obj.useNrm };
    },
  };
})();
