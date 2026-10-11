/* 用**查看器完全相同的相机与光照模型**在 CPU 上光栅化 GLB，输出 PNG。
 *
 * 为什么必须走到这一步：已经排除了文件损坏、解析失败、取景错误、法线反了，
 * 但用户仍然看到黑屏。继续用"推理"排查没有意义 —— 直接把"查看器应该画出什么"
 * 渲染出来，一眼就能判断：
 *   * 若渲染出来是可见的模型  -> 查看器/浏览器侧的问题（需要用真浏览器验证）
 *   * 若渲染出来也是黑的      -> 光照或颜色模型的问题（数学层面可确定）
 *
 * 复刻的查看器行为（逐条对应 static/viewer.js）：
 *   bbox 中心为 target，radius = 半对角线，dist = 2.8*radius
 *   yaw=0.6 pitch=0.45；proj = perspective(45°, aspect, dist*1e-3, dist*100)
 *   lam = 0.30 + 0.70*max(dot(n, normalize(uLight)), 0)，uLight=(0.4,0.7,0.6)
 *   gl_FragColor = pow(clamp(vCol*lam,0,1), 1/2.2)
 *
 * 用法：node scripts/render_glb_cpu.mjs <file.glb> <out.png> [W] [H]
 */
import fs from 'node:fs';
import zlib from 'node:zlib';

const C = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const ND = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

function parseGlb(buf) {
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const dv = new DataView(ab);
  let off = 12, json = null, bin = null;
  const total = dv.getUint32(8, true);
  while (off + 8 <= Math.min(total, ab.byteLength)) {
    const len = dv.getUint32(off, true), type = dv.getUint32(off + 4, true), s = off + 8;
    if (type === 0x4E4F534A) json = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, s, len)));
    else if (type === 0x004E4942) bin = new Uint8Array(ab, s, len);
    off = s + len;
  }
  return { gltf: json, bin };
}
function accessor(gltf, bin, i, asFloat = false) {
  const a = gltf.accessors[i], n = ND[a.type], Ct = C[a.componentType];
  const bv = gltf.bufferViews[a.bufferView];
  const raw = new Ct(bin.buffer, bin.byteOffset + (bv.byteOffset || 0) + (a.byteOffset || 0), a.count * n);
  if (!asFloat) return { raw, n, count: a.count, normalized: !!a.normalized, ct: a.componentType };
  const out = new Float32Array(a.count * n);
  const scale = a.normalized
    ? (a.componentType === 5121 ? 1 / 255 : a.componentType === 5123 ? 1 / 65535 : 1) : 1;
  for (let i2 = 0; i2 < out.length; i2++) out[i2] = raw[i2] * scale;
  return { raw: out, n, count: a.count, float: true };
}

const [inFile, outFile, Ws = '900', Hs = '650'] = process.argv.slice(2);
const W = parseInt(Ws, 10), H = parseInt(Hs, 10);
const { gltf, bin } = parseGlb(fs.readFileSync(inFile));
const prim = gltf.meshes[0].primitives[0];
const P = accessor(gltf, bin, prim.attributes.POSITION, true).raw;
const Nm = prim.attributes.NORMAL !== undefined ? accessor(gltf, bin, prim.attributes.NORMAL, true).raw : null;
const Col = prim.attributes.COLOR_0 !== undefined
  ? accessor(gltf, bin, prim.attributes.COLOR_0, true).raw : null;
const I = prim.indices !== undefined
  ? accessor(gltf, bin, prim.indices).raw : null;
const nv = P.length / 3, nt = I ? I.length / 3 : nv / 3;
console.log(`顶点 ${nv} 三角形 ${nt}  有法线=${!!Nm} 有颜色=${!!Col}`);

// bbox
let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) {
  const v = P[i * 3 + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v;
}
const ctr = [0, 1, 2].map((k) => (mn[k] + mx[k]) / 2);
const rad = Math.max(1e-6, 0.5 * Math.hypot(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]));
const yaw = 0.6, pitch = 0.45, dist = rad * 2.8;
const eye = [
  ctr[0] + dist * Math.cos(pitch) * Math.sin(yaw),
  ctr[1] + dist * Math.sin(pitch),
  ctr[2] + dist * Math.cos(pitch) * Math.cos(yaw),
];
const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot = (a, b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const nr = (a) => { const l = Math.hypot(...a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; };
const zA = nr(sub(eye, ctr)), xA = nr(cross([0, 1, 0], zA)), yA = cross(zA, xA);
const aspect = W / H, f = 1 / Math.tan((Math.PI / 4) / 2);
const LINE = nr([0.4, 0.7, 0.6]);

// 投影所有顶点
const sx = new Float32Array(nv), sy = new Float32Array(nv), sz = new Float32Array(nv);
for (let i = 0; i < nv; i++) {
  const p = sub([P[i*3], P[i*3+1], P[i*3+2]], eye);
  const vz = dot(p, zA), vx = dot(p, xA), vy = dot(p, yA);
  const ze = -vz;
  sz[i] = ze;
  const nx = (f / aspect) * vx / (ze || 1e-9), ny = f * vy / (ze || 1e-9);
  sx[i] = (nx * 0.5 + 0.5) * W;
  sy[i] = (1 - (ny * 0.5 + 0.5)) * H;
}

// z-buffer 光栅化
const zbuf = new Float32Array(W * H).fill(Infinity);
const img = new Uint8Array(W * H * 3);
for (let t = 0; t < nt; t++) {
  const a = I ? I[t*3] : t*3, b = I ? I[t*3+1] : t*3+1, c = I ? I[t*3+2] : t*3+2;
  if (sz[a] <= 0 || sz[b] <= 0 || sz[c] <= 0) continue;
  const x0 = sx[a], y0 = sy[a], x1 = sx[b], y1 = sy[b], x2 = sx[c], y2 = sy[c];
  const minX = Math.max(0, Math.floor(Math.min(x0, x1, x2)));
  const maxX = Math.min(W - 1, Math.ceil(Math.max(x0, x1, x2)));
  const minY = Math.max(0, Math.floor(Math.min(y0, y1, y2)));
  const maxY = Math.min(H - 1, Math.ceil(Math.max(y0, y1, y2)));
  if (minX > maxX || minY > maxY) continue;
  const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
  if (Math.abs(area) < 1e-9) continue;
  const inv = 1 / area;
  for (let py = minY; py <= maxY; py++) {
    for (let px = minX; px <= maxX; px++) {
      const cx = px + 0.5, cy = py + 0.5;
      let w0 = ((x1 - x0) * (cy - y0) - (cx - x0) * (y1 - y0)) * inv;
      let w1 = ((cx - x0) * (y2 - y0) - (x2 - x0) * (cy - y0)) * inv;
      const w2 = 1 - w0 - w1;
      if (w0 < 0 || w1 < 0 || w2 < 0) continue;
      const invz = w2 / sz[a] + w1 / sz[b] + w0 / sz[c];
      const z = 1 / invz;
      const o = py * W + px;
      if (z >= zbuf[o]) continue;
      zbuf[o] = z;
      // 顶点色：用重心插值（注意顶点顺序 a,b,c 对应 w2,w1,w0）
      let r = 0.7, g = 0.72, bl = 0.75;
      if (Col) {
        r = w2*Col[a*3] + w1*Col[b*3] + w0*Col[c*3];
        g = w2*Col[a*3+1] + w1*Col[b*3+1] + w0*Col[c*3+1];
        bl = w2*Col[a*3+2] + w1*Col[b*3+2] + w0*Col[c*3+2];
      }
      let lam = 1.0;
      if (Nm) {
        const nx3 = w2*Nm[a*3] + w1*Nm[b*3] + w0*Nm[c*3];
        const ny3 = w2*Nm[a*3+1] + w1*Nm[b*3+1] + w0*Nm[c*3+1];
        const nz3 = w2*Nm[a*3+2] + w1*Nm[b*3+2] + w0*Nm[c*3+2];
        const l = Math.hypot(nx3, ny3, nz3) || 1;
        const d = Math.max(0, (nx3/l)*LINE[0] + (ny3/l)*LINE[1] + (nz3/l)*LINE[2]);
        lam = 0.30 + 0.70 * d;
      }
      const gam = (v) => Math.round(255 * Math.pow(Math.max(0, Math.min(1, v * lam)), 1 / 2.2));
      img[o*3] = gam(r); img[o*3+1] = gam(g); img[o*3+2] = gam(bl);
    }
  }
}

// 统计
let lit = 0, sum = 0;
for (let i = 0; i < W * H; i++) if (zbuf[i] !== Infinity) { lit++; sum += img[i*3]; }
console.log(`覆盖像素 ${lit}/${W*H} = ${(lit/(W*H)*100).toFixed(1)}%  平均 R=${(sum/Math.max(1,lit)).toFixed(1)}`);

// 写 PNG（RGBA，无滤波）
const raw = Buffer.alloc((W * 4 + 1) * H);
for (let y = 0; y < H; y++) {
  raw[y * (W * 4 + 1)] = 0;
  for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 3, p = y * (W * 4 + 1) + 1 + x * 4;
    const solid = zbuf[y * W + x] !== Infinity;
    raw[p] = solid ? img[o] : 24;
    raw[p+1] = solid ? img[o+1] : 26;
    raw[p+2] = solid ? img[o+2] : 30;
    raw[p+3] = 255;
  }
}
const crcT = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1); t[n] = c; }
  return t;
})();
function crc32(b) { let c = 0xFFFFFFFF; for (let i = 0; i < b.length; i++) c = crcT[(c ^ b[i]) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td), 0);
  return Buffer.concat([len, td, crc]);
}
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4);
ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
fs.writeFileSync(outFile, Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
  chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0)),
]));
console.log('已写出 ' + outFile);
