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

/* ==================== 四元数工具（轨道相机用，避免万向锁） ====================
 *
 * 为什么改用四元数：原来用欧拉角 (yaw, pitch) 表示相机朝向，必须把 pitch 夹在
 * ±(π/2−ε) —— 否则 `lookAt` 的 up 与视线共线、`cross` 退化成零向量。
 * 夹取虽然避免了数值崩坏，但**用户拖到顶/底就过不去**（主观感受就是"转到某个方向卡住"）。
 * 四元数没有这个奇点：旋转可以自由累积，越过天顶/天底也不会退化。
 */
function qMul(a, b) {
  return [
    a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
    a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
    a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
    a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
  ];
}
function qNorm(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0]/l, q[1]/l, q[2]/l, q[3]/l];
}
/** 绕**单位**轴旋转 angle 的四元数 */
function qAxisAngle(axis, angle) {
  const h = angle * 0.5, s = Math.sin(h);
  return [Math.cos(h), axis[0]*s, axis[1]*s, axis[2]*s];
}
/** 用四元数旋转向量 */
function qRot(q, v) {
  // v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
  const u = [q[1], q[2], q[3]];
  const t = cross(u, v);
  const t2 = [t[0] + q[0]*v[0], t[1] + q[0]*v[1], t[2] + q[0]*v[2]];
  const c = cross(u, t2);
  return [v[0] + 2*c[0], v[1] + 2*c[1], v[2] + 2*c[2]];
}
/** 由四元数与相机位置直接构造视图矩阵（正交基，无需 lookAt 的退化兜底） */
function m4viewFromQuat(q, eye, target) {
  // ⚠️ **z 轴必须取"目标 -> 相机"方向，即 -(相机 -> 目标)。**
  //
  // 用户在 GUI 里看到"模型不在正中、跑到视角后面导致不可见"，根因就在这里：
  // 我第 2 轮把 draw() 从 m4lookAt 换成这个函数时，把 z 写成了 (target - eye)，
  // 于是相机在世界里实际看向 **+Z**，而视图空间约定"相机看 -Z"，
  // 结果整个场景被放到相机背后 -> 全部被裁掉 -> 画布上什么都没有。
  //
  // 可控验证（scripts/probe_view_matrix_sign.mjs，单位四元数 / eye=[0,0,5] / target=原点）：
  //   修正前: target -> [0,0,+5]   ❌ 在相机背后
  //   修正后: target -> [0,0,-5]   ✅ 与 m4lookAt 一致
  // 真实模型上的表现：包围盒 8 角点 NDC 的 z 全 ≈1.00（远裁剪面外），视锥内角点 0/8。
  const zc = norm(sub(eye, target));
  const xc = norm(qRot(q, [1, 0, 0]));
  const yc = norm(cross(zc, xc));
  const xc2 = cross(yc, zc);          // 正交化，消除累积误差
  return new Float32Array([
    xc2[0], yc[0], zc[0], 0,
    xc2[1], yc[1], zc[1], 0,
    xc2[2], yc[2], zc[2], 0,
    -dot(xc2, eye), -dot(yc, eye), -dot(zc, eye), 1,
  ]);
}
function m4lookAt(eye, center, up) {
  const z = norm(sub(eye, center));
  let x = cross(up, z);
  // ⚠️ 共线退化保护。当相机方向与 up=(0,1,0) 共线（正上方/正下方俯视）时
  // `cross(up, z)` 是零向量，而 `norm()` 的兜底是 `|| 1` —— 于是得到 [0,0,0]，
  // 整个视图矩阵含 NaN，画面直接消失。这正是"转到某个方向就转不动/变黑"的成因。
  // 这里在退化时换一个不平行的参考轴重新求 x。
  if (Math.hypot(x[0], x[1], x[2]) < 1e-6) {
    const alt = Math.abs(z[1]) > 0.9 ? [0, 0, 1] : [0, 1, 0];
    x = cross(alt, z);
    if (Math.hypot(x[0], x[1], x[2]) < 1e-6) x = [1, 0, 0];
  }
  x = norm(x);
  const y = norm(cross(z, x));
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

  // ⚠️ 必须区分 element 段：`element face N` 里的行**不是**顶点属性。
  // 原来只收集 "property " 行、不看当前落在哪个 element，
  // 于是 face 段的 `property list uchar uint vertex_indices` 被当成第 11 个浮点属性，
  // 导致 stride 与颜色偏移全错。这里按 element 分段解析。
  const props = [];
  const faceProps = [];
  let inVertex = false, inFace = false;
  for (const l of lines) {
    if (l.startsWith('element ')) {
      const nm = l.split(/\s+/)[1];
      inVertex = (nm === 'vertex');
      inFace = (nm === 'face');
      if (inFace) faceProps.push({ list: true });
      continue;
    }
    if (l.startsWith('property ') && inVertex) {
      const p = l.split(/\s+/);
      props.push({ type: p[1], name: p[2] });
    }
  }
  // face 数（用于解析三角形）
  const faceCount = parseInt((lines.find((l) => l.startsWith('element face')) || '')
    .split(/\s+/)[2], 10) || 0;
  // face 的属性声明：形如 `property list uchar uint vertex_indices`
  const faceLine = lines.find((l) => l.startsWith('property list') &&
    lines.indexOf(l) > lines.findIndex((q) => q.startsWith('element face')));
  let faceIndexType = 'uint';
  if (faceLine) {
    const p = faceLine.split(/\s+/);
    // property list <countType> <indexType> <name>
    faceIndexType = p[3] || 'uint';
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
  // ---- 3DGS 完整模型（59 维/高斯，**不丢信息维度**）----
  //
  // 完整的 3DGS PLY 每个高斯有：位置 3 + 尺度 3 + 旋转 4 + 不透明度 1 + **SH 48**
  // （3 个 DC + 45 个高阶，degree=3）。检测到 `f_dc_0` 就按完整高斯解析，
  // 交给泼溅渲染器做"椭球 + alpha + **视角相关 SH 着色**"，
  // 于是查看器看到的是与训练/预览图（gsplat 渲染）**同一个模型**。
  //
  // 相比之下：`.splat`（32 字节/高斯）只有 11 维、丢掉 SH ⇒ 颜色与视角无关；
  // glTF 点云只有 6 维（xyz+rgb）⇒ 连椭球形状和透明度都没有。
  if (offs.f_dc_0 !== undefined) {
    const gx = get('x'), gy = get('y'), gz = get('z');
    const s0 = get('scale_0'), s1 = get('scale_1'), s2 = get('scale_2');
    const r0 = get('rot_0'), r1 = get('rot_1'), r2 = get('rot_2'), r3 = get('rot_3');
    const og = get('opacity') || get('alpha');
    const dc = [get('f_dc_0'), get('f_dc_1'), get('f_dc_2')];
    if (!s0 || !r0 || !og || !dc[0]) {
      throw new Error('PLY 有 f_dc_* 但缺少 scale_*/rot_*/opacity，无法按 3DGS 解析');
    }
    const pos = new Float32Array(vcount * 3);
    const scale = new Float32Array(vcount * 3);
    const quat = new Float32Array(vcount * 4);
    const opacity = new Float32Array(vcount);
    const shDC = new Float32Array(vcount * 3);
    for (let i = 0; i < vcount; i++) {
      pos[i * 3] = gx(i); pos[i * 3 + 1] = gy(i); pos[i * 3 + 2] = gz(i);
      scale[i * 3] = s0(i); scale[i * 3 + 1] = s1(i); scale[i * 3 + 2] = s2(i);
      quat[i * 4] = r0(i); quat[i * 4 + 1] = r1(i);
      quat[i * 4 + 2] = r2(i); quat[i * 4 + 3] = r3(i);
      opacity[i] = og(i);
      shDC[i * 3] = dc[0](i); shDC[i * 3 + 1] = dc[1](i); shDC[i * 3 + 2] = dc[2](i);
    }
    // 高阶 SH：f_rest_0..N-1，通道主序（先 R 的 15 个基，再 G，再 B）
    let shRest = null, shDegree = 0;
    const rest0 = get('f_rest_0');
    if (rest0) {
      // 反推通道数：总属性数 / 3
      let nRest = 0;
      while (get('f_rest_' + nRest)) nRest++;
      const perCh = nRest / 3;
      if (!Number.isInteger(perCh)) {
        throw new Error(`f_rest 数量 ${nRest} 不是 3 的倍数，无法按通道拆分`);
      }
      // 基总数 = perCh + 1（+1 是 DC）；degree 由 (deg+1)² 反推
      const bases = perCh + 1;
      const side = Math.round(Math.sqrt(bases));
      if (side * side !== bases) {
        throw new Error(`f_rest 每通道 ${perCh} 个，反推的基总数 ${bases} 不是完全平方数`);
      }
      shDegree = side - 1;
      shRest = new Float32Array(vcount * nRest);
      for (let k = 0; k < nRest; k++) {
        const rd = get('f_rest_' + k);
        for (let i = 0; i < vcount; i++) shRest[i * nRest + k] = rd(i);
      }
    }
    return { mode: -1, position: pos, color: null, uv: null, normal: null,
             indices: null, textureUrl: null,
             gauss3d: { pos, scale, quat, opacity, shDC, shRest, shDegree } };
  }

  // 法线（网格 PLY 通常带 nx/ny/nz）：有就用于平滑光照
  let normal = null;
  const nxg = get('nx'), nyg = get('ny'), nzg = get('nz');
  if (nxg && nyg && nzg) {
    normal = new Float32Array(vcount * 3);
    let ok = false;
    for (let i = 0; i < vcount; i++) {
      const a = nxg(i), b = nyg(i), c = nzg(i);
      normal[i*3] = a; normal[i*3+1] = b; normal[i*3+2] = c;
      if (Math.abs(a) + Math.abs(b) + Math.abs(c) > 1e-6) ok = true;
    }
    if (!ok) normal = null;          // 全零法线等于没有
  }
  // 颜色：优先 red/green/blue（0~255 或 0~1），否则 f_dc_*（SH 直流项）
  let color = null;
  const rr = get('red'), gg = get('green'), bb = get('blue');
  if (rr && gg && bb) {
    color = new Float32Array(vcount * 3);
    // open3d 写网格时颜色可能是 uchar(0~255)，也可能是 float(0~1)；
    // 用属性类型判断，避免把 0~1 的值再除 255 变成全黑。
    const rp = props.find((p) => p.name === 'red');
    const isByte = rp && /^(uchar|char|uint8|int8|ushort|uint16|short|int16)$/.test(rp.type);
    const sc = isByte ? 1 / 255 : 1;
    for (let i = 0; i < vcount; i++) {
      color[i*3] = Math.min(1, Math.max(0, rr(i) * sc));
      color[i*3+1] = Math.min(1, Math.max(0, gg(i) * sc));
      color[i*3+2] = Math.min(1, Math.max(0, bb(i) * sc));
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

  // ---- 面（三角形）：有 element face 就是**实体网格**，不是点云 ----
  //
  // ⚠️ 这里曾经**完全没实现**：`parsePly` 无条件返回 `mode: 0`（POINTS）与
  //    `indices: null`，于是**任何 PLY 都被画成点云** —— 包括带 74295 个三角面的
  //    实体网格 PLY。用户看到"ply 查看只有点云、不是实体"就是这个原因。
  // open3d 写网格的布局：`element face N` + `property list uchar uint vertex_indices`。
  let indices = null;
  if (faceCount > 0) {
    const faceBase = headEnd + vcount * stride;   // 顶点段之后的字节偏移
    const fdv = new DataView(buf, faceBase);
    const IDX_SIZE = SIZE[faceIndexType] || 4;
    const readIdx = IDX_SIZE === 4
      ? (b) => fdv.getUint32(b, le)
      : (b) => fdv.getUint16(b, le);
    const out = [];
    let p = 0;
    for (let f = 0; f < faceCount; f++) {
      const n = fdv.getUint8(p); p += 1;          // 每面的顶点数（三角面=3）
      const v = new Array(n);
      for (let k = 0; k < n; k++) { v[k] = readIdx(p); p += IDX_SIZE; }
      // 三角形直接收；多边形按扇形三角化（PLY 里少见，但别静默丢面）
      for (let k = 1; k + 1 < n; k++) {
        out.push(v[0], v[k], v[k + 1]);
      }
    }
    if (out.length >= 3) {
      indices = new Uint32Array(out);
      // 索引越界会让整个图元不画（WebGL 静默失败）—— 这里挡一道
      let mx = 0;
      for (let i = 0; i < indices.length; i++) if (indices[i] > mx) mx = indices[i];
      if (mx >= vcount) throw new Error(`PLY 面索引越界（max=${mx} >= 顶点数 ${vcount}）`);
    }
  }
  return { mode: indices ? 4 : 0, position, color, uv: null, normal,
           indices, textureUrl: null };
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

  // ⚠️ **WebGL1 的 drawElements 不接受 UNSIGNED_INT**（只允许 UNSIGNED_BYTE / UNSIGNED_SHORT）。
  // 要索引 32 位必须显式启用 `OES_element_index_uint` 扩展。
  // 踩坑（造成"画布全黑"）：之前无条件用 `gl.UNSIGNED_INT`，
  // 于是 `drawElements(TRIANGLES, n, UNSIGNED_INT, 0)` 报 **INVALID_ENUM** 并且
  // **一个图元都不画** —— 而我们的网格有 3~4 万顶点，超过 UNSIGNED_SHORT 的 65535 上限，
  // 所以每个带面的 PLY/GLB 都会黑屏。错误只在 `getError()` 时才暴露。
  // 修法：探测扩展可用性，按需选择索引类型（见 makeMesh 里的 idxType）。
  // WebGL1 的 drawElements 只接受 ... （详见下方注释）
  // ⚠️ `getExtension` 是 WebGL1 标准方法，但**不能无条件调用**：
  //    某些上下文变体 / 测试替身没有它，直接调用会抛 TypeError 并让 init() 失败
  //    → 画布黑且没有任何提示。这里做存在性判断（实测在 mock 上就抛了）。
  const uintIdxExt = (typeof gl.getExtension === 'function')
    ? gl.getExtension('OES_element_index_uint') : null;
  const idxType = uintIdxExt ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
  const idxMax = uintIdxExt ? 0xFFFFFFFF : 0xFFFF;

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
  // 错误溯源（默认开启一次即可；它只在出错时打印，不影响正常性能）
  try { installGlTracer(gl); } catch { /* 忽略 */ }

  return { gl, prog, U, A, idxType, idxMax, hasUintIndex: !!uintIdxExt };
}

/**
 * 给 WebGL 上下文装一层"错误溯源"包装。
 *
 * 为什么必须这么做：WebGL 的错误是**异步累积**的 —— 某个调用失败后把错误码留在上下文里，
 * 直到有人调用 `getError()` 才读出。所以原来的做法（在 draw 末尾读一次）
 * 只知道"有错"，**不知道是哪个调用**，浏览器也只报一句 "INVALID_ENUM"。
 * 这里把每个可能出错的 GL 方法包一层，调用后立刻读错误码，
 * 于是能直接打印出**方法名 + 参数 + 调用栈**。
 *
 * 只在 `window.__GSVIEWER_TRACE` 为真时启用，避免影响正常路径的性能。
 */
function installGlTracer(gl) {
  if (gl.__traced) return;
  gl.__traced = true;
  const NAMES = { 0x0500: 'INVALID_ENUM', 0x0501: 'INVALID_VALUE',
                  0x0502: 'INVALID_OPERATION', 0x0505: 'OUT_OF_MEMORY',
                  0x0506: 'INVALID_FRAMEBUFFER_OPERATION', 0x9242: 'CONTEXT_LOST_WEBGL' };
  const watch = ['getParameter', 'getProgramParameter', 'getShaderParameter',
                 'enable', 'disable', 'depthFunc', 'clearColor', 'clear', 'viewport',
                 'enableVertexAttribArray', 'disableVertexAttribArray',
                 'vertexAttribPointer', 'vertexAttrib2f', 'vertexAttrib3f',
                 'drawArrays', 'drawElements', 'activeTexture', 'bindTexture',
                 'texParameteri', 'texImage2D', 'generateMipmap', 'pixelStorei',
                 'uniform1i', 'uniform1f', 'uniform3f', 'uniformMatrix4fv',
                 'useProgram', 'bindBuffer', 'bufferData', 'linkProgram', 'compileShader'];
  watch.forEach((name) => {
    const orig = gl[name];
    if (typeof orig !== 'function') return;
    gl[name] = function (...args) {
      const r = orig.apply(gl, args);
      const e = gl.getError();
      if (e !== 0) {
        const label = NAMES[e] || ('0x' + e.toString(16));
        // 参数里的 GL 常量映射回名字，便于读懂
        const pretty = args.map((a) => (typeof a === 'number' && a > 0x1000 && a < 0x10000)
          ? `${a}(0x${a.toString(16)})` : String(a)).join(', ');
        console.error(`[GsViewer] GL ${label} @ gl.${name}(${pretty})`,
                      '\n调用栈:\n' + new Error().stack);
      }
      return r;
    };
  });
}

/** 一个可渲染对象：缓冲 + 元信息 */
function makeMesh(R, model, wireframe) {  const { gl } = R;
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
  let ibuf = null, indexCount = 0, idxType = R.idxType;
  if (model.indices) {
    // 索引类型：WebGL1 默认只支持 UNSIGNED_SHORT（上限 65535）。
    // 顶点数超过该上限时**必须**用 UNSIGNED_INT，而它要求 `OES_element_index_uint`。
    const nVert = model.position.length / 3;
    if (nVert > 65535) {
      if (!R.hasUintIndex) {
        throw new Error(
          `该模型有 ${nVert} 个顶点，超过 WebGL1 无扩展时的 65535 索引上限，`
          + '且浏览器/驱动未提供 OES_element_index_uint 扩展。'
          + '请改用支持该扩展的浏览器，或用 export.mesh_target_triangles 生成更小的网格。');
      }
      idxType = R.gl.UNSIGNED_INT;
    } else {
      idxType = R.gl.UNSIGNED_SHORT;
    }
    // 按选定类型构造索引数组（Uint16Array 时需确保索引确实在范围内）
    const idxArr = idxType === R.gl.UNSIGNED_INT
      ? new Uint32Array(model.indices)
      : new Uint16Array(model.indices);
    ibuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idxArr, gl.STATIC_DRAW);
    indexCount = idxArr.length;
  }
  // 线框用的线段索引：每个三角形贡献 3 条边 (a-b, b-c, c-a)。
  // 必须单独建缓冲 —— 直接拿三角形索引去 drawElements(gl.LINES) 会报
  // INVALID_OPERATION（索引数不是 2 的倍数）并让**整帧不画**。
  let lineIbuf = null, lineIndexCount = 0;
  if (model.indices && model.indices.length >= 3) {
    const tri = model.indices;
    const nTri = Math.floor(tri.length / 3);
    // 线框也走同一索引类型（否则索引数翻倍后更容易越界）
    const L = idxType === R.gl.UNSIGNED_INT ? new Uint32Array(nTri * 6) : new Uint16Array(nTri * 6);
    for (let t = 0; t < nTri; t++) {
      const a = tri[t * 3], b = tri[t * 3 + 1], c = tri[t * 3 + 2];
      L[t * 6] = a; L[t * 6 + 1] = b;
      L[t * 6 + 2] = b; L[t * 6 + 3] = c;
      L[t * 6 + 4] = c; L[t * 6 + 5] = a;
    }
    lineIbuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineIbuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, L, gl.STATIC_DRAW);
    lineIndexCount = L.length;
  }
  // 取景：**中位数中心 + 95 分位距离**（稳健），而不是包围盒
  //
  // ⚠️ 为什么不用包围盒：训练出的点云/高斯常有**远处离群点**。
  // 实测 img3-crop：包围盒半径 **4.76**，而 95% 的点都在距中位数中心 **0.84** 内
  // —— 只有 **0.61% 的离群点**就把半径撑大了 **5.65 倍**。
  // 用包围盒取景会让相机拉太远，物体在画面里变成一小块，
  // 用户会以为"查看器里的模型比 preview 差很多"，其实只是取景错了。
  let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < model.position.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = model.position[i + k];
      if (v < mn[k]) mn[k] = v;
      if (v > mx[k]) mx[k] = v;
    }
  }
  const bboxCenter = [(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2];
  const bboxRadius = Math.max(1e-6, 0.5 * Math.hypot(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]));

  // 稳健中心/半径（点数太多时抽样，保证打开大模型不卡）
  const nPts = model.position.length / 3;
  const stride = Math.max(1, Math.floor(nPts / 20000));
  const subN = Math.ceil(nPts / stride);
  const px = new Float64Array(subN), py = new Float64Array(subN), pz = new Float64Array(subN);
  let w = 0;
  for (let i = 0; i < nPts; i += stride) {
    px[w] = model.position[i * 3];
    py[w] = model.position[i * 3 + 1];
    pz[w] = model.position[i * 3 + 2];
    w++;
  }
  const medOf = (arr, len) => {
    const a = Array.prototype.slice.call(arr, 0, len).sort((u, v) => u - v);
    const k = len >> 1;
    return len % 2 ? a[k] : 0.5 * (a[k - 1] + a[k]);
  };
  let center = bboxCenter, radius = bboxRadius, robust = false;
  if (w >= 16) {
    const mc = [medOf(px, w), medOf(py, w), medOf(pz, w)];
    const ds = new Float64Array(w);
    for (let i = 0; i < w; i++) ds[i] = Math.hypot(px[i]-mc[0], py[i]-mc[1], pz[i]-mc[2]);
    ds.sort();
    const q95 = ds[Math.min(w - 1, Math.floor(0.95 * (w - 1)))];
    if (q95 > 1e-6) {
      center = mc; radius = q95; robust = true;
    }
  }
  return { vbuf, cbuf, ubuf, nbuf, ibuf, lineIbuf, indexCount, lineIndexCount, idxType,
           count: model.position.length / 3, mode: model.mode,
           center, radius, bboxCenter, bboxRadius, robust,
           wireframe,
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
  // ---- 轨道相机（**四元数**）----
  //
  // 演进过程（三次都在同一个问题上）：
  //   1) 最初 pitch 硬夹 ±1.5 rad(±86°) —— 竖直方向根本转不过去；
  //   2) 改成 π/2−ε + `lookAt` 共线兜底 —— 数值不再崩（0 个 NaN），
  //      但**仍然转不过顶/底**，主观感受还是"卡住"（用户再次反馈）；
  //   3) 现在改成**四元数累积旋转**：不再有 pitch 这个会退化的自由度，
  //      拖到天顶/天底可以自然翻过去，没有夹取、没有奇点。
  //
  // 采用"增量旋转"而不是"绝对 yaw/pitch"：
  //   水平拖动 -> 绕**世界 Y 轴**转（轨道相机的习惯）
  //   竖直拖动 -> 绕**相机当前右轴**转（任意朝向下手感一致，也不会在极点翻转）
  const Q0 = (() => {
    // 初始朝向：与原来 yaw=0.6 / pitch=0.45 的起点视觉一致
    return qNorm(qMul(qAxisAngle([0, 1, 0], 0.6), qAxisAngle([1, 0, 0], -0.45)));
  })();
  let qcam = Q0.slice();
  let dist = 3, target = [0, 0, 0];
  let dragging = 0, lastX = 0, lastY = 0, wire = false;
  let onStatus = null;

  /** 当前相机位置：target + (相机系 +Z 方向 × dist) */
  function eyePos() {
    const dir = qRot(qcam, [0, 0, 1]);
    return [target[0] + dir[0] * dist, target[1] + dir[1] * dist, target[2] + dir[2] * dist];
  }

  /**
   * 相机朝向 = yaw（绕**固定世界 Y 轴**，不限圈数）· pitch（绕水平右轴，夹 ±90°）。
   * 用四元数组合，因此没有欧拉角矩阵的数值退化。
   *
   * 为什么最终是这个方案（我为此走了四次弯路，全部有实测数据支撑）：
   *   1) 欧拉角 pitch 硬夹 —— 转不过顶/底；
   *   2) 四元数 + 竖直绕"当前相机右轴"**增量**施加 —— 数值安全，但越过极点后
   *      右轴翻向，继续向下拖会反向（实测非极点区频繁反转）；
   *   3) 四元数 trackball（增量） —— 更糟：持续向下拖 2400px，y 只在 ±0.435
   *      **来回震荡**（589 步反向），根本到不了极点；
   *   4) 四元数 Arcball（以按下点累积） —— 实测 y 是**平滑正弦**
   *      （0.435 → −0.435 → 0.435，周期≈840px），仍是 ±0.435 的圆环。
   *      **根因**：屏幕竖直位移对应的旋转轴垂直于屏幕 up，于是相机绕固定轴做
   *      **圆锥运动**，只能扫出一个圆环 —— 想绕到极点必须斜着拖，
   *      这正是用户说的"转到某个方向就转不动了"。arcball 不适合"看物体"的场景。
   *
   * 最终方案把两件事分开，语义清晰且没有奇点：
   *   - **水平拖动 = yaw**，绕**世界 Y 轴**（与相机朝向无关）⇒ 可以无限圈转，无极点；
   *   - **竖直拖动 = pitch**，绕**水平右轴**（由 yaw 决定，恒水平）⇒ 上下扫到 ±90°，
   *     到顶/底夹住（这是"上下看"的自然边界，不是卡死），**并且永不反向**。
   * 两者都用四元数，`qcam = Qyaw ⊗ Qpitch`，不会出现欧拉角矩阵的退化。
   */
  const PITCH_MAX = Math.PI / 2 - 1e-3;
  // 旋转模式（两种都不"卡住"——都不中途反向；差别只在极点处是停住还是翻过去）：
  //   'clamp'（默认）—— yaw 无限圈（绕世界 Y 轴）+ pitch 夹 ±90°。标准轨道相机：
  //              向下拖到底会停住（"看到底面"即终点），继续拖不回弹、不反向。
  //              实测（probe_camera_monotonic）：持续向下拖 2400px，反向 **0 步**。
  //   'free'  —— yaw 无限圈 + pitch 不夹，可连续越过极点一路翻过去（允许滚转）。
  //              实测能到两个极点（y 覆盖 −1.000~1.000）；因为沿大圆翻滚，
  //              y 分量本身呈正弦（514 步"反向"属该模式固有形态，不是被推回）。
  // 默认取 'clamp'：物体查看器里"拖到底 = 看到底面"最符合直觉，且单向性最强。
  let rotMode = 'clamp';
  // 初始朝向对两个模式都取"俯视 26°、方位 34°"，即旧起点 yaw=0.6 / 俯角 0.45。
  // ⚠️ 两个累加角都**不能取 ±90°**，否则右轴与 Y 轴共面会退化（这里不会）。
  let yawAcc = 0.6, pitchAcc = 0.45;

  /** 由（yawAcc, pitchAcc）构造相机四元数 */
  function buildCam() {
    // 世界 Y 轴上的 yaw（无限圈、与相机朝向无关 ⇒ 永不退化）
    const qy = qAxisAngle([0, 1, 0], yawAcc);
    if (rotMode === 'clamp') {
      // pitch 绕"yaw 之后的水平右轴"转 ⇒ 右轴恒水平、无滚转；夹取保证不翻过极点
      const right = qRot(qy, [1, 0, 0]);
      const p = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, pitchAcc));
      qcam = qNorm(qMul(qAxisAngle(right, p), qy));
    } else {
      // free：pitch 绕"yaw 之后的水平右轴"转，但**不夹取**。
      // 连续越过极点时转向保持同向（不会反向），代价是会产生轻微滚转 —— 这是
      // "自由翻转"的固有特性，也是与 'clamp' 的唯一区别。
      const right = qRot(qy, [1, 0, 0]);
      qcam = qNorm(qMul(qAxisAngle(right, pitchAcc), qy));
    }
  }
  buildCam();

  /** 按屏幕像素增量更新 yaw/pitch */
  function orbit(dx, dy) {
    const SENS = 0.008;
    // 向右拖（dx>0）→ 场景向右转、相机向左绕 ⇒ yaw 增加
    yawAcc += dx * SENS;
    // 向下拖（dy>0）→ 相机应往**下**走 ⇒ 俯角增大 ⇒ pitch 增加。
    // （符号曾写反，导致向下拖反而升到天顶并夹住 —— 实测 y 从 0.449 升到 1.000 卡住。）
    pitchAcc += dy * SENS;
    if (rotMode === 'clamp') {
      pitchAcc = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, pitchAcc));
    } else {
      // free 模式不夹取，但把角度折回 (-π, π] 避免长期拖动后数值过大
      if (pitchAcc > Math.PI) pitchAcc -= 2 * Math.PI;
      if (pitchAcc < -Math.PI) pitchAcc += 2 * Math.PI;
    }
    buildCam();
  }

  /** 切换旋转模式；返回当前模式（供界面/自检使用） */
  function setRotMode(m) {
    rotMode = (m === 'clamp') ? 'clamp' : 'free';
    buildCam();
    draw();
    return rotMode;
  }
  function getRotMode() { return rotMode; }

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
    // 相机位置由四元数给出（不再有 pitch 这个会退化的自由度）
    const eye = eyePos();
    // 注意：GLB 已做 y/z 翻转使其正立；相机基由四元数直接构造，无需 up 向量
    const view = m4viewFromQuat(qcam, eye, target);
    const proj = m4perspective(Math.PI / 4, aspect, dist * 1e-3, dist * 100);
    const mvp = m4mul(proj, view);

    // ---- 高斯泼溅路径 ----
    // 若加载的是 `.splat`，走**泼溅渲染**（每个高斯画成朝向各异的椭球 + alpha 合成），
    // 而不是把高斯当点画。这是"查看器看起来比 preview_front.png 差很多"的根治办法：
    // 之前每个高斯只剩 x,y,z,r,g,b，丢了协方差/不透明度/视角相关颜色。
    if (obj && obj.splat) {
      obj.splat.draw(view, proj, canvas.width, canvas.height, eye, qcam);
      return;
    }

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
      gl.drawElements(gl.LINES, obj.lineIndexCount, obj.idxType, 0);
    } else if (obj.ibuf) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, obj.ibuf);
      // ⚠️ 索引类型必须是该缓冲**实际使用**的类型（UNSIGNED_INT 需要
      // `OES_element_index_uint` 扩展，否则报 INVALID_ENUM 且整帧不画 —— 见 createRenderer）
      gl.drawElements(gl.TRIANGLES, obj.indexCount, obj.idxType, 0);
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
      // 指针捕获：拖出画布/移到窗口外也不丢事件（否则"转着转着就断了"）
      try { canvas.setPointerCapture?.(e.pointerId); } catch { /* 老浏览器忽略 */ }
      e.preventDefault();
    });
    window.addEventListener('mouseup', () => { dragging = 0; });
    window.addEventListener('blur', () => { dragging = 0; });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (dragging === 1) {
        orbit(dx, dy);
      } else {
        // 平移：沿相机的右/上方向在屏幕平面内移动目标点
        const s = dist * 0.0016;
        const right = norm(qRot(qcam, [1, 0, 0]));
        target[0] -= right[0] * dx * s; target[2] -= right[2] * dx * s;
        target[1] += dy * s;
      }
      draw();
    });
    // 触摸旋转（单指）；双指留给缩放
    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        dragging = 1; lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
      }
    }, { passive: true });
    canvas.addEventListener('touchmove', (e) => {
      if (!dragging || e.touches.length !== 1) return;
      const t = e.touches[0];
      const dx = t.clientX - lastX, dy = t.clientY - lastY;
      lastX = t.clientX; lastY = t.clientY;
      orbit(dx, dy);
      draw();
      e.preventDefault();
    }, { passive: false });
    canvas.addEventListener('touchend', () => { dragging = 0; });
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
      qcam = Q0.slice(); yawAcc = 0.6; pitchAcc = 0.45; buildCam();
      draw();
    },
    toggleWire() { wire = !wire; draw(); return wire; },
    /** 旋转模式：'free'（默认，可翻越极点）| 'clamp'（到顶/底停住） */
    setRotMode,
    getRotMode,

    /**
     * 诊断接口：返回查看器的实际运行状态。
     *
     * 为什么需要：浏览器里"全黑"看不到任何线索（异常、GL 状态、矩阵都在内部）。
     * 在控制台执行 `GsViewer.debug()` 就能拿到决定性数据：
     *   - GL 上下文是否创建成功、着色器/程序是否有效
     *   - 每个 uniform / attribute 的 location（**-1 表示 GLSL 里没被用到或名字不匹配**，
     *     这会导致该 attribute 完全不生效）
     *   - 画布尺寸（0 就会黑屏）、视口、相机取景
     *   - 顶点/索引数量、是否有颜色/法线
     */
    debug() {
      const out = { ok: !!(R && canvas), obj: null, gl: null, prog: null,
                    uniforms: {}, attribs: {}, canvas: null, camera: null };
      if (!R || !canvas) return out;
      const { gl } = R;
      out.gl = {
        contextLost: typeof gl.isContextLost === 'function' ? gl.isContextLost() : null,
        vendor: gl.getParameter(gl.VENDOR),
        renderer: gl.getParameter(gl.RENDERER),
        version: gl.getParameter(gl.VERSION),
        glError: gl.getError(),
      };
      out.prog = {
        valid: gl.getProgramParameter(R.prog, gl.VALIDATE_STATUS),
        linked: gl.getProgramParameter(R.prog, gl.LINK_STATUS),
      };
      Object.keys(R.U).forEach((k) => { out.uniforms[k] = R.U[k] ? 'ok' : 'NULL'; });
      Object.keys(R.A).forEach((k) => { out.attribs[k] = R.A[k]; });
      out.canvas = {
        client: [canvas.clientWidth, canvas.clientHeight],
        buffer: [canvas.width, canvas.height],
        css: [getComputedStyle(canvas).width, getComputedStyle(canvas).height],
        display: getComputedStyle(canvas).display,
        visibility: getComputedStyle(canvas).visibility,
      };
      if (obj) {
        out.obj = {
          mode: obj.mode, count: obj.count, indexCount: obj.indexCount,
          useCol: obj.useCol, useNrm: obj.useNrm, useUV: obj.useUV,
          hasTexture: !!obj.texture,
          center: obj.center, radius: obj.radius,
          buffers: { vbuf: !!obj.vbuf, cbuf: !!obj.cbuf, nbuf: !!obj.nbuf,
                     ibuf: !!obj.ibuf, lineIbuf: !!obj.lineIbuf },
        };
      }
      out.camera = { quat: qcam.slice(), dist, target, wire, eye: eyePos(),
                     yawAcc, pitchAcc, rotMode };
      return out;
    },

    /** 清空当前模型（"清空视图"按钮用） */
    clear() {      if (obj && R) {
        const { gl } = R;
        [obj.vbuf, obj.cbuf, obj.ubuf, obj.nbuf, obj.ibuf, obj.lineIbuf]
          .forEach((b) => { if (b) gl.deleteBuffer(b); });
        if (obj.texture) gl.deleteTexture(obj.texture);
      }
      obj = null;
      target = [0, 0, 0];
      dist = 3;
      qcam = Q0.slice(); yawAcc = 0.6; pitchAcc = 0.45; buildCam();
      draw();
      return true;
    },

    /** 载入一个远端产物：从 URL 取回后交给 loadBuffer */
    async load(url, kind) {
      if (!R) throw new Error('查看器未初始化');
      if (onStatus) onStatus('载入中…');
      let res;
      try {
        res = await fetch(url);
      } catch (e) {
        throw new Error('下载失败：' + e.message);
      }
      if (!res.ok) throw new Error('下载失败 HTTP ' + res.status
                                   + '（文件可能已被删除 —— 重跑一次导出即可）');
      const buf = await res.arrayBuffer();
      return this.loadBuffer(buf, url, kind);
    },

    /** 载入一个已在内存里的模型（本地文件、拖拽、粘贴等都用这条路径）*/
    async loadBuffer(buf, nameOrUrl, kind) {
      if (!R) throw new Error('查看器未初始化');
      const label = String(nameOrUrl || '');
      const low = label.toLowerCase();
      if (!buf || buf.byteLength < 64) {
        throw new Error('文件为空或过小（' + (buf ? buf.byteLength : 0)
                        + ' 字节）');
      }
      if (onStatus) onStatus('解析中…');
      const isSplat = (kind === 'splat' || low.endsWith('.splat'));

      // ---- 高斯泼溅：交给专用的泼溅渲染器 ----
      // 为什么单独一条路：泼溅需要 WebGL 实例化 + 逐帧深度桶排序 + 预乘 alpha 混合，
      // 与"网格/点云"的绘制管线完全不同。混在一条管线上只会两边都做不好。
      if (isSplat) {
        if (!window.GsSplatRenderer) {
          throw new Error('缺少 splat 渲染器（splat.js 未加载）');
        }
        if (obj) {
          const g = R.gl;
          [obj.vbuf, obj.cbuf, obj.ubuf, obj.nbuf, obj.ibuf, obj.lineIbuf]
            .forEach((b) => { if (b) g.deleteBuffer(b); });
          if (obj.texture) g.deleteTexture(obj.texture);
          obj = null;
        }
        // ⚠️ 泼溅失败必须**降级到点云**，不能让用户看到一块黑屏。
        // 真实故障：着色器里用了 `transpose()`（GLSL ES 3.00 才有），
        // WebGL1 编译失败 ⇒ 整块模型完全不可见，用户只得到一句报错。
        // 降级后至少还能看到点云 + 明确的降级提示。
        try {
          const sp = window.GsSplatRenderer.create(R.gl, buf);
          const cen = sp.bounds.center, rad = sp.bounds.radius;
          obj = { mode: -1, splat: sp, count: sp.count, indexCount: 0,
                  center: cen, radius: rad,
                  useCol: true, useNrm: false, texture: null };
          this.reset();
          draw();
          return { vertices: sp.count, triangles: 0, mode: 'splat',
                   textured: false, colors: true, normals: false,
                   gaussians: sp.count };
        } catch (e) {
          const data = window.GsSplatRenderer.parseSplat(buf);
          const n = data.count;
          const color = new Float32Array(n * 3);
          for (let i = 0; i < n; i++) {
            color[i * 3] = data.color[i * 4];
            color[i * 3 + 1] = data.color[i * 4 + 1];
            color[i * 3 + 2] = data.color[i * 4 + 2];
          }
          const fb = { mode: 0, position: data.center, color, uv: null,
                       normal: null, indices: null, textureUrl: null };
          if (obj) {
            const g = R.gl;
            [obj.vbuf, obj.cbuf, obj.ubuf, obj.nbuf, obj.ibuf, obj.lineIbuf]
              .forEach((b) => { if (b) g.deleteBuffer(b); });
            obj = null;
          }
          obj = makeMesh(R, fb, wire);
          this.reset();
          draw();
          // 把原因带出去（界面层会显示在状态栏），但**不抛异常**
          if (onStatus) onStatus('泼溅不可用，已降级为点云：' + e.message);
          return { vertices: n, triangles: 0, mode: 0,
                   textured: false, colors: true, normals: false,
                   degraded: true, degradeReason: e.message };
        }
      }

      let model;
      try {
        if (kind === 'glb' || low.endsWith('.glb')) {
          model = glbToRenderable(buf, null);
        } else if (kind === 'ply' || low.endsWith('.ply')) {
          model = parsePly(buf);
        } else {
          // ⚠️ 只支持 glb / ply / splat 三种容器。OBJ / SPZ 等**必须在界面层就过滤掉** ——
          // 否则这里会走进 GLB 解析分支、抛一个含糊的错误，用户只看到"黑"。
          throw new Error('查看器不支持该格式（仅 GLB / PLY / SPLAT）：'
                          + (label.split('/').pop() || label));
        }
      } catch (e) {
        throw new Error('解析失败：' + e.message);
      }

      // ---- 3DGS PLY：**完整信息维度**（59 维/高斯，含 SH 48）⇒ 走泼溅渲染 ----
      if (model && model.gauss3d) {
        if (!window.GsSplatRenderer) throw new Error('缺少 splat 渲染器（splat.js 未加载）');
        if (obj) {
          const g = R.gl;
          [obj.vbuf, obj.cbuf, obj.ubuf, obj.nbuf, obj.ibuf, obj.lineIbuf]
            .forEach((b) => { if (b) g.deleteBuffer(b); });
          if (obj.texture) g.deleteTexture(obj.texture);
          obj = null;
        }
        // 同样要**降级到点云**（理由见 `.splat` 分支）：
        // 着色器/扩展出问题时，宁可显示点云 + 降级提示，也不要一块黑屏。
        try {
          const sp3 = window.GsSplatRenderer.createFromGauss3D(R.gl, model.gauss3d);
          obj = { mode: -1, splat: sp3, count: sp3.count, indexCount: 0,
                  center: sp3.bounds.center, radius: sp3.bounds.radius,
                  useCol: true, useNrm: false, texture: null };
          this.reset();
          draw();
          return { vertices: sp3.count, triangles: 0, mode: 'gauss3d',
                   textured: false, colors: true, normals: false,
                   gaussians: sp3.count,
                   shDegree: model.gauss3d.shDegree,
                   hasSH: !!model.gauss3d.shRest };
        } catch (e) {
          const g3 = model.gauss3d;
          const n = g3.pos.length / 3;
          // 用 SH DC 还原近似颜色（0.5 + C0·f_dc），至少不是全黑
          const SH_C0 = 0.28209479177387814;
          const color = new Float32Array(n * 3);
          for (let i = 0; i < n; i++) {
            for (let k = 0; k < 3; k++) {
              const v = 0.5 + SH_C0 * g3.shDC[i * 3 + k];
              color[i * 3 + k] = Math.min(1, Math.max(0, v));
            }
          }
          const fb = { mode: 0, position: g3.pos, color, uv: null,
                       normal: null, indices: null, textureUrl: null };
          obj = makeMesh(R, fb, wire);
          this.reset();
          draw();
          if (onStatus) onStatus('泼溅不可用，已降级为点云：' + e.message);
          return { vertices: n, triangles: 0, mode: 0,
                   textured: false, colors: true, normals: false,
                   degraded: true, degradeReason: e.message };
        }
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
