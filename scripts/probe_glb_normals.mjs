/* 检查 GLB 顶点法线的**朝向**：是否与"从质心指向顶点"一致（即朝外）。
 *
 * 为什么查这个：`model_mesh.glb` 在查看器里是黑的，但已排除
 *   - 文件损坏（magic=glTF、HTTP 200、字节完整）
 *   - 解析失败（Node 能解出 24273 顶点 / 79724 三角形）
 *   - 取景错误（100% 顶点落在视口内、深度在 near/far 之间）
 * 剩下的可能就是**光照恒为 0**：顶点色本身只有 0.26，若法线朝内，
 * `diff = max(dot(n, L), 0) = 0`，只剩 0.30 的环境项 ⇒ 0.26*0.30 ≈ 0.078 ⇒ 近乎全黑。
 *
 * 判据：
 *   1) 顶点法线与"质心->顶点"方向的平均点积（>0 朝外）
 *   2) 顶点法线与其相邻**面法线**的平均点积（三角形用 CCW 时面法线朝外）
 *
 * 用法：node scripts/probe_glb_normals.mjs <file.glb>
 */
import fs from 'node:fs';

const C = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const N = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

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
  const a = gltf.accessors[i], n = N[a.type], Ct = C[a.componentType];
  const bv = gltf.bufferViews[a.bufferView];
  return new Ct(bin.buffer, bin.byteOffset + (bv.byteOffset || 0) + (a.byteOffset || 0), a.count * n);
}

const f = process.argv[2];
const { gltf, bin } = parse(fs.readFileSync(f));
const prim = gltf.meshes[0].primitives[0];
const P = acc(gltf, bin, prim.attributes.POSITION);
const nv = P.length / 3;
console.log(`${f.split(/[\\/]/).pop()}: 顶点 ${nv}`);
if (prim.attributes.NORMAL === undefined) {
  console.log('  **没有 NORMAL 属性**（查看器只能退回固定方向，光照是错的）');
  process.exit(0);
}
const Nm = acc(gltf, bin, prim.attributes.NORMAL);

// 质心
const ctr = [0, 0, 0];
for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) ctr[k] += P[i * 3 + k] / nv;

// 1) 顶点法线 vs 径向
let dR = 0, nR = 0;
for (let i = 0; i < nv; i++) {
  const r = [P[i*3]-ctr[0], P[i*3+1]-ctr[1], P[i*3+2]-ctr[2]];
  const rl = Math.hypot(...r) || 1;
  const nl = Math.hypot(Nm[i*3], Nm[i*3+1], Nm[i*3+2]) || 1;
  dR += (r[0]/rl)*(Nm[i*3]/nl) + (r[1]/rl)*(Nm[i*3+1]/nl) + (r[2]/rl)*(Nm[i*3+2]/nl);
  nR++;
}
console.log(`  顶点法线·径向（质心->顶点）平均 = ${(dR/nR).toFixed(4)}   ${dR/nR > 0 ? '朝外 OK' : '**朝内（光照会全黑）**'}`);

// 2) 顶点法线 vs 面法线（从索引算，CCW 为正面）
if (prim.indices !== undefined) {
  const I = acc(gltf, bin, prim.indices);
  const nt = Math.floor(I.length / 3);
  let sum = 0, cnt = 0;
  const step = Math.max(1, Math.floor(nt / 20000));
  for (let t = 0; t < nt; t += step) {
    const a = I[t*3], b = I[t*3+1], c = I[t*3+2];
    const e1 = [P[b*3]-P[a*3], P[b*3+1]-P[a*3+1], P[b*3+2]-P[a*3+2]];
    const e2 = [P[c*3]-P[a*3], P[c*3+1]-P[a*3+1], P[c*3+2]-P[a*3+2]];
    const fn = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
    const fl = Math.hypot(...fn) || 1;
    for (const v of [a, b, c]) {
      const nl = Math.hypot(Nm[v*3], Nm[v*3+1], Nm[v*3+2]) || 1;
      sum += (fn[0]/fl)*(Nm[v*3]/nl) + (fn[1]/fl)*(Nm[v*3+1]/nl) + (fn[2]/fl)*(Nm[v*3+2]/nl);
      cnt++;
    }
  }
  console.log(`  顶点法线·面法线平均 = ${(sum/cnt).toFixed(4)}   ${sum/cnt > 0 ? '一致 OK' : '**相反（面片朝向与法线矛盾）**'}`);
}

// 3) 模拟查看器光照能拿到多少亮度（当前顶点色很暗，这一步很关键）
if (prim.attributes.COLOR_0 !== undefined) {
  const Col = acc(gltf, bin, prim.attributes.COLOR_0);
  const norm = !!gltf.accessors[prim.attributes.COLOR_0].normalized;
  const sc = norm && gltf.accessors[prim.attributes.COLOR_0].componentType === 5121 ? 1/255 : 1;
  let mr = 0, mg = 0, mb = 0;
  for (let i = 0; i < nv; i++) { mr += Col[i*3]*sc/nv; mg += Col[i*3+1]*sc/nv; mb += Col[i*3+2]*sc/nv; }
  const L = [0.4, 0.7, 0.6];
  const Ll = Math.hypot(...L);
  // 用各顶点法线估平均漫反射
  let diff = 0;
  for (let i = 0; i < nv; i++) {
    const nl = Math.hypot(Nm[i*3], Nm[i*3+1], Nm[i*3+2]) || 1;
    diff += Math.max(0, (Nm[i*3]/nl)*L[0]/Ll + (Nm[i*3+1]/nl)*L[1]/Ll + (Nm[i*3+2]/nl)*L[2]/Ll) / nv;
  }
  const lam = 0.30 + 0.70 * diff;
  console.log(`  顶点色均值 = [${mr.toFixed(3)}, ${mg.toFixed(3)}, ${mb.toFixed(3)}]`);
  console.log(`  平均漫反射项 = ${diff.toFixed(4)}  => 光照系数 lam = ${lam.toFixed(3)}`);
  console.log(`  预计屏幕亮度 ≈ [${(mr*lam).toFixed(3)}, ${(mg*lam).toFixed(3)}, ${(mb*lam).toFixed(3)}]`
              + `  (sRGB 后 ≈ ${Math.round(255*Math.pow(mr*lam, 1/2.2))})`);
}
