/* 精确定位 `getParameter` / `getProgramParameter` 等**按 pname 分派**的 GL 调用
 * 是否传了非法常量 —— 这是 WebGL 报 INVALID_ENUM 的唯一途径。
 *
 * 为什么单独做这个：我加的 `GsViewer.debug()` 里调用了
 *   gl.getParameter(gl.VENDOR/RENDERER/VERSION) 与
 *   gl.getProgramParameter(prog, gl.VALIDATE_STATUS)
 * 如果常量名写错或浏览器不支持，就会报 INVALID_ENUM，
 * 但**不影响渲染**。必须先确认这条错误的归属，否则会误导排查方向。
 *
 * 用法: node scripts/check_gl_pnames.mjs
 */
import fs from 'node:fs';

// WebGL1 getParameter 合法 pname 全集（Khronos IDL 节选：字符串类 + 数值类常用项）
const GETPARAM_OK = new Set([
  // 字符串类
  'VENDOR', 'RENDERER', 'VERSION', 'SHADING_LANGUAGE_VERSION',
  // 数值类（常用）
  'MAX_TEXTURE_SIZE', 'MAX_CUBE_MAP_TEXTURE_SIZE', 'MAX_VERTEX_ATTRIBS',
  'MAX_VERTEX_UNIFORM_VECTORS', 'MAX_VARYING_VECTORS',
  'MAX_COMBINED_TEXTURE_IMAGE_UNITS', 'MAX_VERTEX_TEXTURE_IMAGE_UNITS',
  'MAX_TEXTURE_IMAGE_UNITS', 'MAX_FRAGMENT_UNIFORM_VECTORS',
  'MAX_VIEWPORT_DIMS', 'ALIASED_POINT_SIZE_RANGE', 'ALIASED_LINE_WIDTH_RANGE',
  'SUBPIXEL_BITS', 'RED_BITS', 'GREEN_BITS', 'BLUE_BITS', 'ALPHA_BITS',
  'DEPTH_BITS', 'STENCIL_BITS', 'SAMPLE_BUFFERS', 'SAMPLES',
  'CURRENT_PROGRAM', 'ARRAY_BUFFER_BINDING', 'ELEMENT_ARRAY_BUFFER_BINDING',
  'FRAMEBUFFER_BINDING', 'RENDERBUFFER_BINDING', 'TEXTURE_BINDING_2D',
  'TEXTURE_BINDING_CUBE_MAP', 'VIEWPORT', 'SCISSOR_BOX',
  'COLOR_CLEAR_VALUE', 'COLOR_WRITEMASK', 'DEPTH_CLEAR_VALUE', 'DEPTH_FUNC',
  'DEPTH_RANGE', 'DEPTH_WRITEMASK', 'STENCIL_CLEAR_VALUE', 'STENCIL_FUNC',
  'STENCIL_REF', 'STENCIL_VALUE_MASK', 'STENCIL_WRITEMASK', 'STENCIL_FAIL',
  'STENCIL_PASS_DEPTH_FAIL', 'STENCIL_PASS_DEPTH_PASS', 'BLEND_COLOR',
  'BLEND_EQUATION_RGB', 'BLEND_EQUATION_ALPHA', 'BLEND_DST_RGB', 'BLEND_SRC_RGB',
  'BLEND_DST_ALPHA', 'BLEND_SRC_ALPHA', 'CULL_FACE_MODE', 'FRONT_FACE',
  'LINE_WIDTH', 'POLYGON_OFFSET_FACTOR', 'POLYGON_OFFSET_UNITS',
  'GENERATE_MIPMAP_HINT', 'UNPACK_ALIGNMENT', 'PACK_ALIGNMENT',
  'IMPLEMENTATION_COLOR_READ_TYPE', 'IMPLEMENTATION_COLOR_READ_FORMAT',
  'COMPRESSED_TEXTURE_FORMATS', 'ACTIVE_TEXTURE',
  'UNPACK_FLIP_Y_WEBGL', 'UNPACK_PREMULTIPLY_ALPHA_WEBGL',
  'UNPACK_COLORSPACE_CONVERSION_WEBGL', 'MAX_RENDERBUFFER_SIZE',
]);

// getProgramParameter 合法 pname
const GETPROGPARAM_OK = new Set([
  'DELETE_STATUS', 'LINK_STATUS', 'VALIDATE_STATUS', 'ATTACHED_SHADERS',
  'ACTIVE_ATTRIBUTES', 'ACTIVE_UNIFORMS', 'CURRENT_PROGRAM',
]);

// getShaderParameter 合法 pname
const GETSHADERPARAM_OK = new Set(['SHADER_TYPE', 'DELETE_STATUS', 'COMPILE_STATUS']);

const RULES = [
  [/\bgl\.getParameter\(\s*gl\.([A-Z][A-Z0-9_]*)/g, GETPARAM_OK, 'getParameter'],
  [/\bgl\.getProgramParameter\([^,]+,\s*gl\.([A-Z][A-Z0-9_]*)/g, GETPROGPARAM_OK, 'getProgramParameter'],
  [/\bgl\.getShaderParameter\([^,]+,\s*gl\.([A-Z][A-Z0-9_]*)/g, GETSHADERPARAM_OK, 'getShaderParameter'],
];

const files = process.argv.slice(2).length
  ? process.argv.slice(2)
  : ['src/gsflow/web/static/viewer.js', 'src/gsflow/web/static/app.js'];

let bad = 0;
for (const f of files) {
  const lines = fs.readFileSync(f, 'utf8').split(/\r?\n/);
  lines.forEach((line, i) => {
    for (const [re, ok, fn] of RULES) {
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(line)) !== null) {
        if (!ok.has(m[1])) {
          console.log(`  [非法 pname] ${f}:${i + 1}  ${fn}(gl.${m[1]})`);
          console.log(`         ${line.trim().slice(0, 120)}`);
          bad++;
        }
      }
    }
  });
}
console.log(bad ? `\n发现 ${bad} 处可能触发 INVALID_ENUM 的调用`
                : '\n✅ getParameter / getProgramParameter / getShaderParameter 的 pname 均合法');
process.exitCode = bad ? 1 : 0;
