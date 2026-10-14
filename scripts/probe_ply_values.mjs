/* 按 viewer.js 的 parsePly 逻辑读 PLY，报告位置/颜色/索引范围。
 * 用于回答"查看器全黑"：位置若为垃圾值 -> 相机取景失效 -> 什么都看不到。
 *
 * 用法: node scripts/probe_ply_values.mjs <file.ply>
 */
import fs from 'node:fs';

const f = process.argv[2];
const buf = fs.readFileSync(f);

// 找 end_header
let headEnd = -1, headText = '';
for (let j = 0; j < Math.min(buf.length, 65536) - 4; j++) {
  if (buf[j] === 0x65 && buf[j+1] === 0x6E && buf[j+2] === 0x64 &&
      buf[j+3] === 0x5F) {
    let k = j, s = '';
    while (k < buf.length && buf[k] !== 0x0A) { s += String.fromCharCode(buf[k]); k++; }
    if (s.startsWith('end_header')) { headEnd = k + 1; headText = buf.slice(0, headEnd).toString('latin1'); break; }
  }
}
if (headEnd < 0) throw new Error('找不到 end_header');
const lines = headText.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
const fmt = (lines.find(l => l.startsWith('format')) || '').split(/\s+/)[1] || '';
const le = fmt.includes('little');
const vcount = parseInt((lines.find(l => l.startsWith('element vertex')) || '').split(/\s+/)[2], 10);
const faceCount = parseInt((lines.find(l => l.startsWith('element face')) || '').split(/\s+/)[2], 10) || 0;
console.log(`format=${fmt} le=${le} vertex=${vcount} face=${faceCount}`);

const props = [];
let inVertex = false;
for (const l of lines) {
  if (l.startsWith('element ')) { inVertex = (l.split(/\s+/)[1] === 'vertex'); continue; }
  if (inVertex && l.startsWith('property ')) {
    const p = l.split(/\s+/);
    props.push({ type: p[1], name: p[2] });
  }
}
const SIZE = { char:1, uchar:1, int8:1, uint8:1, short:2, ushort:2, int16:2, uint16:2,
               int:4, uint:4, int32:4, uint32:4, float:4, float32:4, double:8, float64:8 };
let stride = 0; const offs = {};
for (const p of props) { offs[p.name] = stride; stride += SIZE[p.type] || 4; }
console.log(`stride=${stride} props=${props.map(p => p.name + ':' + p.type).join(',')}`);

const dv = new DataView(buf.buffer, buf.byteOffset + headEnd);
function mk(name) {
  const p = props.find(q => q.name === name);
  if (!p) return null;
  const t = p.type, o = offs[name];
  return (i) => {
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
}
const rx = mk('x'), ry = mk('y'), rz = mk('z');
const rr = mk('red'), gg = mk('green'), bb = mk('blue');
const nx = mk('nx'), ny = mk('ny'), nz = mk('nz');

let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < vcount; i++) {
  const v = [rx(i), ry(i), rz(i)];
  for (let k = 0; k < 3; k++) { if (v[k] < mn[k]) mn[k] = v[k]; if (v[k] > mx[k]) mx[k] = v[k]; }
}
const ctr = [0,1,2].map(k => (mn[k]+mx[k])/2);
const radius = Math.max(1e-6, 0.5 * Math.hypot(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]));
console.log(`位置 min=[${mn.map(v=>v.toFixed(3))}] max=[${mx.map(v=>v.toFixed(3))}]`);
console.log(`中心=[${ctr.map(v=>v.toFixed(3))}] 半径=${radius.toFixed(4)}  dist=2.8*r=${(radius*2.8).toFixed(3)}`);
console.log(`非有限位置数=${(() => { let n=0; for (let i=0;i<vcount;i++) if (!isFinite(rx(i))||!isFinite(ry(i))||!isFinite(rz(i))) n++; return n; })()}`);

if (rr) {
  let c0 = 0, c1 = 0, c2 = 0;
  for (let i = 0; i < vcount; i++) { c0 += rr(i); c1 += gg(i); c2 += bb(i); }
  console.log(`颜色原始均值=[${(c0/vcount).toFixed(2)}, ${(c1/vcount).toFixed(2)}, ${(c2/vcount).toFixed(2)}]` +
              `（uchar 应 0~255；若 <=1 说明是 float 0~1，除 255 会变黑）`);
  const rp = props.find(p => p.name === 'red');
  console.log(`red 属性类型=${rp.type}`);
}
if (nx) {
  let z = 0; for (let i = 0; i < vcount; i++) if (Math.abs(nx(i))+Math.abs(ny(i))+Math.abs(nz(i)) < 1e-9) z++;
  console.log(`零法线数=${z}/${vcount}`);
}
// 面索引
if (faceCount > 0) {
  const faceLine = lines.find(l => l.startsWith('property list'));
  const it = faceLine ? faceLine.split(/\s+/)[3] : 'uint';
  const IS = SIZE[it] || 4;
  const fdv = new DataView(buf.buffer, buf.byteOffset + headEnd + vcount * stride);
  const rd = IS === 4 ? (b) => fdv.getUint32(b, le) : (b) => fdv.getUint16(b, le);
  let p = 0, mxIdx = 0, nTri = 0;
  for (let k = 0; k < faceCount; k++) {
    const n = fdv.getUint8(p); p += 1;
    const v = [];
    for (let j = 0; j < n; j++) { v.push(rd(p)); p += IS; }
    for (let j = 0; j < n; j++) if (v[j] > mxIdx) mxIdx = v[j];
    if (n >= 3) nTri += n - 2;
  }
  console.log(`索引类型=${it} 最大索引=${mxIdx}（应 < ${vcount}）三角形数=${nTri}`);
}
