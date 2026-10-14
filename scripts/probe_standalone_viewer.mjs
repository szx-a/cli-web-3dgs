/* 验证"自包含离线查看器"HTML 真的可用。
 *
 * 为什么必须验：这种"生成一个 HTML"的脚本最容易出现
 *   - 占位符没被替换（里面残留 {xxx}）
 *   - viewer.js 内嵌后与页面脚本冲突（如重复声明、`</script>` 提前闭合）
 *   - base64 数据被截断或转义错误
 * 这些都不会在生成时报错，只有真正打开才暴露 —— 而"打不开/全黑"正是用户反复遇到的问题。
 *
 * 做法：解析 HTML → 抽出内嵌的 viewer.js 与 base64 数据 → 用同一套 GL mock 跑一遍
 *       loadBuffer，确认模型能被解析成网格/点云。
 *
 * 用法: node scripts/probe_standalone_viewer.mjs <standalone.html>
 */
import fs from 'node:fs';
import path from 'node:path';

const file = process.argv[2];
if (!file) { console.error('用法: node scripts/probe_standalone_viewer.mjs <html>'); process.exit(2); }
const html = fs.readFileSync(file, 'utf8');
let bad = 0;
const fail = (m) => { console.log('  **' + m + '**'); bad++; };

console.log(`检查 ${path.basename(file)}  (${(html.length / 1048576).toFixed(2)} MB)`);

// 1) 占位符是否都被替换
const ph = html.match(/\{[a-z_]+\}/g) || [];
const realPh = ph.filter((x) => !['{}'].includes(x) && /^\{(title|viewer_js|model_b64|kind)\}$/.test(x));
if (realPh.length) fail('残留占位符: ' + [...new Set(realPh)].join(', '));
else console.log('  占位符已全部替换');

// 2) 抽出 base64 数据
const m = /const MODEL_B64 = "([A-Za-z0-9+/=]+)";/.exec(html);
if (!m) { fail('找不到内嵌的 MODEL_B64'); }
else {
  const b64 = m[1];
  const buf = Buffer.from(b64, 'base64');
  const magic = buf.slice(0, 4).toString('latin1');
  console.log(`  内嵌数据 ${(buf.length / 1048576).toFixed(2)} MB  magic=${JSON.stringify(magic)}`);
  if (magic !== 'glTF') fail('内嵌数据不是 GLB（magic 应为 glTF）');
  // 3) GLB 结构自检
  if (buf.length > 20) {
    const total = buf.readUInt32LE(8);
    console.log(`  GLB 声明总长 ${total}  实际 ${buf.length}`);
    if (total !== buf.length) fail('GLB 声明长度与实际不符（数据被截断？）');
  }
}

// 4) 抽出内嵌的 viewer.js 并确认可加载
const s = html.indexOf('<script>');
const e = html.indexOf('/* ---- 模型数据');
if (s < 0 || e < 0) fail('找不到内嵌脚本区');
else {
  const inline = html.slice(s + '<script>'.length, e);
  console.log(`  内嵌 viewer.js ${(inline.length / 1024).toFixed(1)} KB`);
  if (inline.includes('</script>')) fail('内嵌 JS 里含 </script>，会提前闭合脚本标签');
  // 关键：window.GsViewer 必须被定义
  const need = ['window.GsViewer', 'loadBuffer', 'setRotMode', 'getRotMode'];
  const miss = need.filter((k) => !inline.includes(k));
  if (miss.length) fail('内嵌 viewer.js 缺少: ' + miss.join(', '));
  else console.log('  GsViewer / loadBuffer / setRotMode 均在');

  // 5) 真正执行一遍（用 GL mock）
  const listeners = {};
  const mkTarget = () => ({
    addEventListener: (t, fn) => { (listeners[t] = listeners[t] || []).push(fn); },
    removeEventListener: () => {}, setPointerCapture: () => {},
  });
  const GLV = {
    DEPTH_TEST: 0x0B71, COLOR_BUFFER_BIT: 0x4000, DEPTH_BUFFER_BIT: 0x0100,
    VERTEX_SHADER: 0x8B31, FRAGMENT_SHADER: 0x8B30, COMPILE_STATUS: 0x8B81,
    LINK_STATUS: 0x8B82, ARRAY_BUFFER: 0x8892, ELEMENT_ARRAY_BUFFER: 0x8893,
    STATIC_DRAW: 0x88E4, FLOAT: 0x1406, UNSIGNED_INT: 0x1405, UNSIGNED_BYTE: 0x1401,
    UNSIGNED_SHORT: 0x1403, RGBA: 0x1908, TEXTURE_2D: 0x0DE1, TEXTURE_WRAP_S: 0x2802,
    TEXTURE_WRAP_T: 0x2803, REPEAT: 0x2901, TEXTURE_MIN_FILTER: 0x2801,
    TEXTURE_MAG_FILTER: 0x2800, LINEAR_MIPMAP_LINEAR: 0x2703, LINEAR: 0x2601,
    UNPACK_FLIP_Y_WEBGL: 0x9240, POINTS: 0, TRIANGLES: 4, LINES: 1, TEXTURE0: 0x84C0,
    VENDOR: 0x1F00, RENDERER: 0x1F01, VERSION: 0x1F02,
  };
  const draws = [];
  function makeGl() {
    let err = 0; const noop = () => {};
    return {
      ...GLV, createShader: () => ({}), shaderSource: noop, compileShader: noop,
      getShaderParameter: () => true, getShaderInfoLog: () => '',
      createProgram: () => ({}), attachShader: noop, linkProgram: noop,
      getProgramParameter: () => true, getProgramInfoLog: () => '', useProgram: noop,
      getUniformLocation: (p, n) => ({ n }),
      getAttribLocation: (p, n) => ({ aPos: 0, aCol: 1, aUV: 2, aNrm: 3 })[n] ?? -1,
      createBuffer: () => ({}), bindBuffer: noop, bufferData: noop, deleteBuffer: noop,
      enableVertexAttribArray: noop, disableVertexAttribArray: noop,
      vertexAttribPointer: noop, vertexAttrib2f: noop, vertexAttrib3f: noop,
      enable: noop, disable: noop, depthFunc: noop, clearColor: noop, clear: noop,
      viewport: noop, uniformMatrix4fv: noop, uniform1f: noop, uniform1i: noop,
      uniform3f: noop, activeTexture: noop, bindTexture: noop, pixelStorei: noop,
      texImage2D: noop, texParameteri: noop, generateMipmap: noop,
      createTexture: () => ({}), deleteTexture: noop,
      drawArrays: (mo, f, c) => draws.push(['drawArrays', mo, c]),
      drawElements: (mo, c, ty) => {
        draws.push(['drawElements', mo, c, ty]);
        if (ty !== GLV.UNSIGNED_SHORT && ty !== GLV.UNSIGNED_BYTE && ty !== GLV.UNSIGNED_INT) {
          err = err || 0x0500;
        }
      },
      getError: () => { const e2 = err; err = 0; return e2; },
      getParameter: () => 4096,
      getExtension: (n) => (n === 'OES_element_index_uint' ? { n } : null),
      isContextLost: () => false,
    };
  }
  const canvas = Object.assign(mkTarget(), {
    clientWidth: 800, clientHeight: 600, width: 800, height: 600, getContext: () => makeGl(),
  });
  const windowStub = Object.assign(mkTarget(), { devicePixelRatio: 1 });
  const documentStub = { getElementById: (id) => (id === 'c' ? canvas : mkTarget()) };
  globalThis.getComputedStyle = () => ({ width: '800px', height: '600px', display: 'block', visibility: 'visible' });
  try {
    new Function('window', 'document', 'getComputedStyle', inline)(
      windowStub, documentStub, globalThis.getComputedStyle);
    if (!windowStub.GsViewer) fail('执行后没有 window.GsViewer');
    else {
      windowStub.GsViewer.init(canvas, () => {});
      const kind = (/const MODEL_KIND = "([a-z]+)";/.exec(html) || [])[1] || 'glb';
      const b64 = (/const MODEL_B64 = "([A-Za-z0-9+/=]+)";/.exec(html) || [])[1];
      const buf = Buffer.from(b64, 'base64');
      const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      const info = await windowStub.GsViewer.loadBuffer(ab, 'inline.' + kind, kind);
      console.log(`  解析结果: 顶点 ${info.vertices}  三角面 ${info.triangles}  mode ${info.mode}` +
                  `  法线=${info.normals}  顶点色=${info.colors}`);
      if (!info.vertices) fail('解析出 0 个顶点');
      if (info.mode === 4 && !info.triangles) fail('是网格模式但没有三角面');
      if (!draws.length) fail('没有产生任何 draw 调用');
      else console.log(`  绘制调用: ${JSON.stringify(draws[0])}`);
    }
  } catch (err) {
    fail('执行内嵌脚本抛错: ' + err.message);
  }
}

console.log('\n结论: ' + (bad ? `**有 ${bad} 项问题**` : '离线查看器可用（双击即可，无需服务）'));
process.exitCode = bad ? 1 : 0;
