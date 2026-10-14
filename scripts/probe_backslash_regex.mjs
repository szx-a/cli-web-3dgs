// 干净地验证 `/\\\\/g` 这个正则到底匹配什么（避免 shell 转义干扰）。
//
// 背景：app.js 行 225 写的是
//     encodeURIComponent(a.replace(/\\\\/g, '/'))
// 而实测 `a.replace(/\\\\/g, '/')` **没有**把 Windows 反斜杠换成 '/'，
// 于是 URL 里出现 %5C -> 服务器 404。
//
// 这里用多种写法对照，确认哪种写法才是"匹配单个反斜杠"。

const s = 'F:\\3dgs-and-ai\\workspace\\job\\01_images_raw';   // 含单个反斜杠的真实字符串

const cases = [
  ['/\\\\/g     (四个反斜杠，源码里的写法)', /\\\\/g],
  ['/\\/g       (两个反斜杠)', /\\/g],
  ['/[\\\\]/g   (字符类)', /[\\]/g],
  ['/[\\\\]{1}/g', /[\\]{1}/g],
];

console.log('待处理字符串（JSON）:', JSON.stringify(s));
console.log('其中反斜杠个数:', (s.match(/\\/g) || []).length);
console.log();
for (const [label, re] of cases) {
  const out = s.replace(re, '/');
  const ok = !out.includes('\\');
  console.log(`${ok ? '[OK]  ' : '[FAIL]'} ${label}`);
  console.log(`       正则: ${String(re)}   source=${JSON.stringify(re.source)}`);
  console.log(`       结果: ${JSON.stringify(out)}`);
}

// 再确认编码后的样子（这正是服务器收到的东西）
console.log('\nencodeURIComponent 之后：');
for (const [label, re] of cases) {
  console.log(`  ${label.padEnd(42)} -> ${encodeURIComponent(s.replace(re, '/')).slice(0, 70)}`);
}

// 从 app.js 源文件里**取字面量**再跑（最接近浏览器的验证）
import fs from 'node:fs';
const src = fs.readFileSync('src/gsflow/web/static/app.js', 'utf8');
const m = /a\.replace\((\/[^,]*\/g)/.exec(src);
if (m) {
  const lit = m[1];
  console.log(`\n从 app.js 取到的正则字面量: ${lit}`);
  const re = eval(lit);
  const out = s.replace(re, '/');
  console.log(`  结果: ${JSON.stringify(out)}`);
  console.log(`  是否仍含反斜杠: ${out.includes('\\') ? '**是（这就是 404 的原因）**' : '否'}`);
}
