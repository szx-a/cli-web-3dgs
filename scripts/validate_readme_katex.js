/**
 * 用真实 KaTeX 编译 README 里每一段公式，找出"退化成文字"的元凶。
 *
 * 之前只能靠规则猜（间距命令、行内 $ 空格），但用户仍反馈"部分公式没渲染"。
 * 这个脚本不再猜：直接 katex.renderToString(..., {throwOnError:true})，
 * 编译失败的段落会连同行号和 KaTeX 的错误信息一起打印。
 *
 * 用法（需要先在工作区装 katex）：
 *   cd env/tmp && npm install katex --cache env/tmp/npm-cache
 *   node scripts/validate_readme_katex.js
 *
 * 说明：GitHub 用 MathJax，KaTeX 更严格。KaTeX 报错的宏在 MathJax 上**可能**能渲染，
 * 但既然用户看到失败，优先按 KaTeX 能过的子集来写是最稳的。
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const README = path.join(ROOT, 'README.md');
const KATEX_PATHS = [
  path.join(ROOT, 'env', 'tmp', 'node_modules', 'katex'),
  path.join(ROOT, 'node_modules', 'katex'),
];

let katex = null;
for (const p of KATEX_PATHS) {
  try {
    katex = require(p);
    break;
  } catch (e) { /* try next */ }
}
if (!katex) {
  console.error('找不到 katex。请先：cd env/tmp && npm install katex --cache env/tmp/npm-cache');
  process.exit(2);
}

const text = fs.readFileSync(README, 'utf8');
const lines = text.split('\n');

function lineOf(index) {
  return text.slice(0, index).split('\n').length;
}

let checked = 0;
const failures = [];

// ---- 块级 $$...$$ ----
const blockRe = /\$\$([\s\S]+?)\$\$/g;
let m;
const blocks = [];
while ((m = blockRe.exec(text)) !== null) {
  blocks.push({ tex: m[1], line: lineOf(m.index), display: true });
}
// ---- 行内 $...$（逐行，跳过代码围栏与已匹配的块）----
let inFence = false;
let masked = text;
for (const b of blocks) {
  masked = masked.slice(0, b.lineStartHint) + 'x'.repeat(0); // no-op, 下面按行处理
}
inFence = false;
const inlineList = [];
for (let i = 0; i < lines.length; i += 1) {
  const ln = lines[i];
  if (ln.trim().startsWith('```')) { inFence = !inFence; continue; }
  if (inFence || ln.includes('$$')) continue;
  const re = /(?<!\$)\$(?!\$)([^$\n]*?)(?<!\$)\$(?!\$)/g;
  let mm;
  while ((mm = re.exec(ln)) !== null) {
    inlineList.push({ tex: mm[1], line: i + 1, display: false });
  }
}

for (const seg of blocks.concat(inlineList)) {
  checked += 1;
  try {
    katex.renderToString(seg.tex, { throwOnError: true, displayMode: seg.display, strict: false });
  } catch (err) {
    failures.push({ ...seg, message: String(err.message || err).split('\n')[0] });
  }
}

console.log(`检查了 ${checked} 段公式（块级 ${blocks.length}、行内 ${inlineList.length}）`);
if (!failures.length) {
  console.log('全部通过 KaTeX 编译 ✔');
  process.exit(0);
}
console.log(`\n${failures.length} 段编译失败：`);
for (const f of failures) {
  console.log(`  [行 ${f.line}] ${f.display ? '块级' : '行内'} — ${f.message}`);
  console.log(`      ${f.tex.trim().replace(/\n/g, ' ').slice(0, 150)}`);
}
process.exit(1);
