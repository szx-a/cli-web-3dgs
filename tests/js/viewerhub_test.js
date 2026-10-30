/**
 * ViewerHub 的行为测试（node 直接跑，无需浏览器）。
 *
 * 覆盖用户报的那条路径：
 *   WebGL 交互（ref）→ 服务端渲染（suspend）→ 返回 WebGL（resume）→ 必须能继续拖拽。
 *
 * 做法：从 app.js 里抽出 `const ViewerHub = (() => { ... })();` 这段源码，
 * 用最小 DOM 桩求值，然后断言 pointer-events / display / active 三者的状态。
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const APP = path.resolve(__dirname, '../../src/gsflow/web/static/app.js');
const src = fs.readFileSync(APP, 'utf8');

// ---- 抽出 ViewerHub 定义（平衡括号，不依赖具体格式） ----
function extractViewerHub(code) {
  const start = code.indexOf('const ViewerHub = (() => {');
  if (start < 0) throw new Error('找不到 ViewerHub 定义');
  let i = code.indexOf('{', start);
  let depth = 0;
  for (let k = i; k < code.length; k += 1) {
    const ch = code[k];
    if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        // 结束位置是 `}`，后面应跟 `)();`
        const tail = code.indexOf(';', k);
        return code.slice(start, tail + 1);
      }
    }
  }
  throw new Error('ViewerHub 定义括号不平衡');
}

// ---- 最小 DOM 桩 ----
function makeEl(name) {
  return {
    id: name,
    style: {},
    dataset: {},
    disabled: false,
    textContent: '',
    title: '',
    removeAttribute() {},
    setAttribute() {},
  };
}

const els = {};
for (const id of ['m-canvas', 'm-image', 'btn-m-wire', 'btn-m-rotmode', 'btn-m-reset', 'm-status', 'm-overlay']) {
  els[id] = makeEl(id);
}
const stage = makeEl('stage');

let redrawCalls = 0;
let refAvailable = true;

const sandbox = {
  console,
  document: {
    getElementById: (id) => els[id] || null,
    querySelector: (sel) => (sel.indexOf('gs-ref-stage') >= 0 ? stage : null),
  },
  window: {
    GsViewer: { redraw() { redrawCalls += 1; }, reset() {}, toggleWire() {}, setRotMode() {}, getRotMode() { return 'free'; } },
    GsRefViewer: { available: () => refAvailable, resetCamera() {} },
  },
};
vm.createContext(sandbox);

const code = extractViewerHub(src) + '\nglobalThis.__ViewerHub = ViewerHub;';
vm.runInContext(code, sandbox);
const Hub = sandbox.__ViewerHub;

let failures = 0;
function check(label, cond, extra) {
  if (cond) {
    console.log(`  [PASS] ${label}`);
  } else {
    failures += 1;
    console.log(`  [FAIL] ${label}${extra ? ' — ' + extra : ''}`);
  }
}

// 1) 高斯模式：ref 层可交互，canvas 不接收事件
Hub.set('ref');
check('set(ref): stage pointer-events=auto', stage.style.pointerEvents === 'auto', String(stage.style.pointerEvents));
check('set(ref): canvas pointer-events=none', els['m-canvas'].style.pointerEvents === 'none', String(els['m-canvas'].style.pointerEvents));
check('set(ref): active=ref', Hub.active === 'ref');

// 2) 点「服务端渲染」→ suspend
Hub.suspend('server');
check('suspend: active=null', Hub.active === null);
check('suspend: stage pointer-events=none', stage.style.pointerEvents === 'none', String(stage.style.pointerEvents));
check('suspend: 记住了挂起前的渲染器', Hub.suspendedKind === 'ref', String(Hub.suspendedKind));

// 3) 点「返回 WebGL」→ resume：必须恢复成 ref 且可交互
const kind = Hub.resume();
check('resume: 返回 ref', kind === 'ref', String(kind));
check('resume: active=ref', Hub.active === 'ref');
check('resume: stage pointer-events 恢复 auto（交互不再被锁）', stage.style.pointerEvents === 'auto', String(stage.style.pointerEvents));
check('resume: canvas 不抢事件', els['m-canvas'].style.pointerEvents === 'none', String(els['m-canvas'].style.pointerEvents));
check('resume: stage 显示', stage.style.display === 'block', String(stage.style.display));
check('resume: 图片层隐藏', els['m-image'].style.display === 'none', String(els['m-image'].style.display));

// 4) glb/网格路径：gs → suspend → resume 必须回到 gs 并重绘
Hub.set('gs');
redrawCalls = 0;
Hub.suspend('image');
check('gs suspend: canvas pointer-events=none', els['m-canvas'].style.pointerEvents === 'none');
const kind2 = Hub.resume();
check('gs resume: 返回 gs', kind2 === 'gs', String(kind2));
check('gs resume: canvas pointer-events=auto', els['m-canvas'].style.pointerEvents === 'auto');
check('gs resume: canvas 显示', els['m-canvas'].style.display === 'block');
check('gs resume: 触发过一次 redraw', redrawCalls === 1, String(redrawCalls));

// 5) 没有可恢复对象时必须干净返回 null（不能崩、也不能假装恢复）
//    场景：一进页就是图片/服务端模式，从未进入过任何 3D 渲染器
Hub.set(null);
Hub.suspend('server');
refAvailable = false;
const kind3 = Hub.resume();
check('无可恢复对象时 resume 返回 null', kind3 === null, String(kind3));

console.log(failures === 0 ? '\n全部通过' : `\n失败 ${failures} 项`);
process.exit(failures === 0 ? 0 : 1);
