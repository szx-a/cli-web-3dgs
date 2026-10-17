/** 导出后端映射为 JSON（供 probe_gui_opacity_params.mjs 读取，避免 Node 直接 spawn Python）。 */
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(import.meta.dirname, '..');
const APP = path.join(ROOT, 'src', 'gsflow', 'web', 'static', 'app.js');
const OUT = path.join(ROOT, 'env', 'tmp', 'opacity_map.json');

// 由调用方先跑 python 生成 JSON；这里只做纯 JS 侧的一致性校验。
const backend = JSON.parse(fs.readFileSync(OUT, 'utf8'));

const src = fs.readFileSync(APP, 'utf8');
const hasRule = src.includes('quantile$') && src.includes('opacity_decay$');
console.log(`app.js 含"数值覆盖"规则: ${hasRule ? 'OK' : '**未找到**'}`);

function isNumericKey(k) {
  return k.endsWith('quantile') || k === 'train.opacity_decay';
}

function guiParams(choice, num, map) {
  const p = { 'train.opacity_cleanup': choice };
  const mapped = map[choice];
  if (!mapped) return p;
  for (const [k, v] of Object.entries(mapped)) {
    p[k] = (isNumericKey(k) && !Number.isNaN(num)) ? String(num) : v;
  }
  return p;
}

let bad = 0;
console.log('\n方案          GUI键数  后端键数  键集合一致?');
console.log('-'.repeat(52));
for (const c of Object.keys(backend)) {
  const g = guiParams(c, Number.NaN, backend);
  const same = Object.keys(g).sort().join(',') === Object.keys(backend[c]).sort().join(',');
  if (!same) bad += 1;
  console.log(`${c.padEnd(13)}${String(Object.keys(g).length).padStart(7)}` +
              `${String(Object.keys(backend[c]).length).padStart(10)}   ${same ? 'OK' : '**不符**'}`);
}

const withNum = guiParams('utilization', 0.07, backend);
const okNum = withNum['train.utilization_prune_quantile'] === '0.07';
console.log(`\n数值覆盖生效（utilization=0.07）: ${okNum ? 'OK' : '**FAIL**'}`);
if (!okNum) bad += 1;

const okDefault = backend.none && Object.keys(backend.none).length === 1;
console.log(`默认方案 none 只标记不打开开关: ${okDefault ? 'OK' : '**FAIL**'}`);
if (!okDefault) bad += 1;

console.log(`\n结论: ${bad === 0 ? 'GUI 与 CLI 参数集合一致' : `**${bad} 项不一致**`}`);
process.exit(bad === 0 ? 0 : 1);
