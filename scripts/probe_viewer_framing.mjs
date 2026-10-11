/* 复算查看器的相机取景：把 GLB 顶点投影到屏幕，统计落在视口内的三角形数。
 *
 * 动机：`model_mesh.glb` 在浏览器里是黑的，但文件本身完好（HTTP 200、magic=glTF、
 * Node 能解析出 24273 顶点 / 79724 三角形）。所以要查"几何到底有没有落在视野里"。
 * 查看器的取景逻辑：
 *     target = bbox 中心;  dist = radius * 2.8
 *     eye = target + dist*(cos(pitch)sin(yaw), sin(pitch), cos(pitch)cos(yaw))
 *     proj = perspective(45°, aspect, dist*1e-3, dist*100)
 * 本脚本用同样的参数把顶点投到 NDC，统计：
 *   - 落在 |x|,|y| <= 1 且 z 在 (-1,1) 内（即真正可见）的顶点比例
 *   - 投影后的包围盒（若整体跑出 [-1,1] 就是取景错了）
 *
 * 用法：node scripts/probe_viewer_framing.mjs <file.glb>
 */
import fs from 'node:fs';

const GLTF_COMP = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const GLTF_ITEMS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

function parse(buf) {
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
function acc(gltf, bin, i) {
  const a = gltf.accessors[i], n = GLTF_ITEMS[a.type], C = GLTF_COMP[a.componentType];
  const bv = gltf.bufferViews[a.bufferView];
  return new C(bin.buffer, bin.byteOffset + (bv.byteOffset || 0) + (a.byteOffset || 0), a.count * n);
}

const inFile = process.argv[2];
const { gltf, bin } = parse(fs.readFileSync(inFile));
const prim = gltf.meshes[0].primitives[0];
const V = acc(gltf, bin, prim.attributes.POSITION);
const I = prim.indices !== undefined ? acc(gltf, bin, prim.indices) : null;
const n = V.length / 3;

// bbox / 中心 / 半径（与 makeMesh 一致）
let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < n; i++) {
  for (let k = 0; k < 3; k++) { const v = V[i * 3 + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
}
const ctr = [0, 1, 2].map((k) => (mn[k] + mx[k]) / 2);
const rad = Math.max(1e-6, 0.5 * Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]));
console.log(`${inFile.split(/[\\/]/).pop()}: 顶点 ${n}  三角形 ${I ? I.length / 3 : '-'}`);
console.log(`  bbox min=[${mn.map((v) => v.toFixed(3))}] max=[${mx.map((v) => v.toFixed(3))}]`);
console.log(`  中心=[${ctr.map((v) => v.toFixed(3))}]  半径=${rad.toFixed(4)}`);

// 查看器默认视角：yaw=0.6 pitch=0.45 dist=2.8*radius
const yaw = 0.6, pitch = 0.45, dist = rad * 2.8;
const target = ctr.slice();
const eye = [
  target[0] + dist * Math.cos(pitch) * Math.sin(yaw),
  target[1] + dist * Math.sin(pitch),
  target[2] + dist * Math.cos(pitch) * Math.cos(yaw),
];
console.log(`  eye=[${eye.map((v) => v.toFixed(3))}] dist=${dist.toFixed(3)}`);

const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot = (a, b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; };

const z = norm(sub(eye, target)), x = norm(cross([0, 1, 0], z)), y = cross(z, x);
const aspect = 800 / 600, fovy = Math.PI / 4, near = dist * 1e-3, far = dist * 100;
const f = 1 / Math.tan(fovy / 2);

let inside = 0, behind = 0, zmin = Infinity, zmax = -Infinity;
let ndcMinX = Infinity, ndcMaxX = -Infinity, ndcMinY = Infinity, ndcMaxY = -Infinity;
for (let i = 0; i < n; i++) {
  const p = [V[i*3]-eye[0], V[i*3+1]-eye[1], V[i*3+2]-eye[2]];
  const vx = dot(p, x), vy = dot(p, y), vz = dot(p, z);
  if (vz >= 0) { behind++; continue; }
  const ze = -vz;
  if (ze < zmin) zmin = ze; if (ze > zmax) zmax = ze;
  const nx = (f / aspect) * vx / ze, ny = f * vy / ze;
  if (nx < ndcMinX) ndcMinX = nx; if (nx > ndcMaxX) ndcMaxX = nx;
  if (ny < ndcMinY) ndcMinY = ny; if (ny > ndcMaxY) ndcMaxY = ny;
  if (Math.abs(nx) <= 1 && Math.abs(ny) <= 1) inside++;
}
console.log(`  相机前方顶点 ${n - behind}/${n}（后方 ${behind}）`);
console.log(`  深度范围 ${zmin.toFixed(3)} ~ ${zmax.toFixed(3)}  (near=${near.toFixed(4)} far=${far.toFixed(1)})`);
console.log(`  NDC 范围 x[${ndcMinX.toFixed(2)}, ${ndcMaxX.toFixed(2)}] y[${ndcMinY.toFixed(2)}, ${ndcMaxY.toFixed(2)}]`);
console.log(`  落在视口内(NDC|x|,|y|<=1)的顶点: ${inside}/${n} = ${(inside/n*100).toFixed(1)}%`);
console.log(inside > n * 0.5 ? '  => 取景正常，几何在视野内'
                             : '  => **取景异常：大部分顶点不在视野内**');
