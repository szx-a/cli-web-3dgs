/* 用 Node 真跑查看器的 GLB 解析逻辑（对真实文件），把"黑屏"定位到具体环节。
 *
 * 为什么这么做：浏览器里只能看到"黑"，看不到异常。而 viewer.js 的解析部分是纯函数，
 * 可以在 Node 里直接跑 —— 于是能明确回答：
 *   解析出来的顶点数/三角形数/mode/有没有颜色/法线/UV 是否正确？
 *
 * 用法：
 *   node scripts/probe_viewer_parse.mjs <file.glb> [file2.glb ...]
 */
import fs from 'node:fs';
import path from 'node:path';

const GLTF_COMP = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array,
                    5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const GLTF_ITEMS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

function parseGlb(buf) {
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const dv = new DataView(ab);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('不是 GLB 文件');
  const total = dv.getUint32(8, true);
  let off = 12, json = null, bin = null;
  while (off + 8 <= Math.min(total, ab.byteLength)) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    const start = off + 8;
    if (type === 0x4E4F534A) json = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, start, len)));
    else if (type === 0x004E4942) bin = new Uint8Array(ab, start, len);
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
  if (!stride || stride === n * Ctor.BYTES_PER_ELEMENT) {
    return { data: new Ctor(bin.buffer, bin.byteOffset + base, count), n, count: acc.count };
  }
  const out = new Ctor(count);
  for (let i = 0; i < acc.count; i++) {
    out.set(new Ctor(bin.buffer, bin.byteOffset + base + i * stride, n), i * n);
  }
  return { data: out, n, count: acc.count };
}

function stats(a) {
  let mn = Infinity, mx = -Infinity, nan = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (Number.isNaN(v)) { nan++; continue; }
    if (v < mn) mn = v; if (v > mx) mx = v;
  }
  return { min: mn, max: mx, nan };
}

for (const f of process.argv.slice(2)) {
  console.log('=== ' + path.basename(f) + ' ===');
  try {
    const buf = fs.readFileSync(f);
    const { gltf, bin } = parseGlb(buf);
    const prim = gltf.meshes[0].primitives[0];
    const attrs = prim.attributes || {};
    console.log('  mode=' + (prim.mode ?? 4) + '  attributes=' + JSON.stringify(attrs)
                + '  indices=' + prim.indices);
    console.log('  bin 大小=' + (bin ? bin.byteLength : 'null'));
    const pos = readAccessor(gltf, bin, attrs.POSITION);
    console.log('  POSITION count=' + pos.count + '  范围 ' + JSON.stringify(stats(pos.data)));
    if (attrs.COLOR_0 !== undefined) {
      const c = readAccessor(gltf, bin, attrs.COLOR_0);
      console.log('  COLOR_0  count=' + c.count + '  ct=' + gltf.accessors[attrs.COLOR_0].componentType
                  + '  normalized=' + !!gltf.accessors[attrs.COLOR_0].normalized
                  + '  范围 ' + JSON.stringify(stats(c.data)));
    }
    if (attrs.NORMAL !== undefined) {
      const n = readAccessor(gltf, bin, attrs.NORMAL);
      console.log('  NORMAL   count=' + n.count + '  范围 ' + JSON.stringify(stats(n.data)));
    }
    if (prim.indices !== undefined) {
      const ix = readAccessor(gltf, bin, prim.indices);
      const nv = pos.count;
      let mx = 0;
      for (let i = 0; i < ix.data.length; i++) if (ix.data[i] > mx) mx = ix.data[i];
      console.log('  INDICES  count=' + ix.count + '  max=' + mx + '  顶点数=' + nv
                  + (mx < nv ? '  越界=否' : '  **越界**'));
    }
    const mat = (gltf.materials || [])[0];
    console.log('  material=' + JSON.stringify(mat && mat.pbrMetallicRoughness));
    console.log('  images=' + JSON.stringify(gltf.images || null));
  } catch (e) {
    console.log('  **解析失败**: ' + e.message);
  }
}
