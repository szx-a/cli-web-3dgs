/**
 * 用 **mathjax-full**（GitHub 的数学渲染引擎）逐段编译文档里的公式。
 *
 * 为什么不用 KaTeX：KaTeX 有扩展别名（例如 `\tfrac` 在 KaTeX 能过、MathJax 报
 * `Undefined control sequence`），用户看到"部分表达式退化成文字"正是这一类。
 *
 * 提取顺序很重要（这是踩过的坑）：
 *   1) 先按行剥掉**围栏代码块**（``` / ~~~，支持长短嵌套与缩进围栏）；
 *   2) 再遮掉**行内代码**（`` `...` ``）——若先遮行内代码，```` ```powershell ```` 里
 *      两个反引号会被配成一对，把围栏破坏掉（docs/03 曾因此误报）；
 *   3) 最后提取 `$$...$$` 与行内 `$...$`（跳过 4 空格缩进的代码块）。
 *
 * 用法：
 *   cd env/tmp && npm install mathjax-full --cache env/tmp/npm-cache
 *   node scripts/validate_readme_mathjax.js [文件路径]
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const TARGET = process.argv[2] ? path.resolve(ROOT, process.argv[2]) : path.join(ROOT, 'README.md');
const MJ_DIRS = [
  path.join(ROOT, 'env', 'tmp', 'node_modules', 'mathjax-full'),
  path.join(ROOT, 'node_modules', 'mathjax-full'),
];

let base = null;
for (const d of MJ_DIRS) {
  if (fs.existsSync(path.join(d, 'js', 'mathjax.js'))) { base = d; break; }
}
if (!base) {
  console.error('找不到 mathjax-full。请先：cd env/tmp && npm install mathjax-full --cache env/tmp/npm-cache');
  process.exit(2);
}

const { mathjax } = require(path.join(base, 'js', 'mathjax.js'));
const { TeX } = require(path.join(base, 'js', 'input', 'tex.js'));
const { liteAdaptor } = require(path.join(base, 'js', 'adaptors', 'liteAdaptor.js'));
const { RegisterHTMLHandler } = require(path.join(base, 'js', 'handlers', 'html.js'));
const { SerializedMmlVisitor } = require(path.join(base, 'js', 'core', 'MmlTree', 'SerializedMmlVisitor.js'));

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: ['base', 'ams', 'newcommand', 'noundefined'] });
const html = mathjax.document('', { InputJax: tex });
const visitor = new SerializedMmlVisitor();

function compile(texString, display) {
  const node = html.convert(texString, { display, em: 16, ex: 8, containerWidth: 1200 });
  return visitor.visitTree(node);
}

const raw = fs.readFileSync(TARGET, 'utf8');
const rawLines = raw.split(/\r?\n/);

// ---- 1) 剥掉围栏代码块（行内容替换为空格，保持行号不变）----
const stripped = [];
let fence = null;
for (const line of rawLines) {
  const clean = line.replace(/\r$/, '');
  const fm = clean.match(/^\s*(`{3,}|~{3,})([^`]*)$/);
  if (fence) {
    if (fm && fm[1][0] === fence.ch && fm[1].length >= fence.len && fm[2].trim() === '') {
      fence = null;
    }
    stripped.push(' '.repeat(clean.length));
    continue;
  }
  if (fm && fm[2].trim() !== '' || (fm && fm[2].trim() === '' && fm[1].length >= 3)) {
    // 开围栏：```lang 或单独 ``` —— 只要不是行内出现
    fence = { ch: fm[1][0], len: fm[1].length };
    stripped.push(' '.repeat(clean.length));
    continue;
  }
  stripped.push(clean);
}

// ---- 2) 遮掉行内代码 ----
const text = stripped.join('\n').replace(/`[^`\n]*`/g, (m) => ' '.repeat(m.length));
const lines = text.split('\n');
const lineOf = (idx) => text.slice(0, idx).split('\n').length;

// ---- 3) 提取公式 ----
const segs = [];
const blockRe = /\$\$([\s\S]+?)\$\$/g;
let m;
while ((m = blockRe.exec(text)) !== null) {
  segs.push({ tex: m[1], line: lineOf(m.index), display: true });
}
for (let i = 0; i < lines.length; i += 1) {
  const ln = lines[i];
  if (ln.includes('$$')) continue;
  const indent = (ln.match(/^\s*/) || [''])[0].length;
  if (indent >= 4) continue; // 缩进代码块
  const re = /(?<!\$)\$(?!\$)([^$\n]*?)(?<!\$)\$(?!\$)/g;
  let mm;
  while ((mm = re.exec(ln)) !== null) segs.push({ tex: mm[1], line: i + 1, display: false });
}

let checked = 0;
const failures = [];
for (const s of segs) {
  checked += 1;
  let mml = '';
  try {
    mml = compile(s.tex, s.display);
  } catch (err) {
    failures.push({ ...s, message: String(err.message || err).split('\n')[0] });
    continue;
  }
  if (mml.includes('<merror') || mml.includes('data-mjx-error')) {
    const msg = (mml.match(/data-mjx-error="([^"]*)"/) || mml.match(/<merror[^>]*>([^<]*)</) || [])[1] || 'merror';
    failures.push({ ...s, message: msg });
  }
}

const rel = path.relative(ROOT, TARGET) || TARGET;
console.log(`${rel}: MathJax 检查了 ${checked} 段（块级 ${segs.filter((s) => s.display).length}、行内 ${segs.filter((s) => !s.display).length}）`);
if (!failures.length) {
  console.log('全部通过 MathJax 编译 ✔');
  process.exit(0);
}
console.log(`\n${failures.length} 段编译失败：`);
for (const f of failures) {
  console.log(`  [行 ${f.line}] ${f.display ? '块级' : '行内'} — ${f.message}`);
  console.log(`      ${f.tex.trim().replace(/\n/g, ' ').slice(0, 160)}`);
}
process.exit(1);
