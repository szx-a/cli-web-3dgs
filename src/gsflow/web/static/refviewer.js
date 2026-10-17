/**
 * 参考实现适配层：用社区主流的 three.js 高斯泼溅渲染库显示模型。
 *
 * ## 为什么不再自己写渲染器
 *
 * 我此前用 WebGL1 **从零手写**了一个泼溅渲染器（逐高斯桶排序、四边形 2σ 近似、
 * 固定低通）。用户反馈得很直接：
 *
 *   > 光栅化你从头写没必要啊，而且也不好，你至少参考一下官方的啊，
 *   > 不然我这里看出来，模型差会误导我
 *
 * 这个批评是对的。一个自研渲染器的差距会被误读成**模型质量差**，
 * 而我没有任何手段证明"差距只来自渲染器"。正确做法是用**已被广泛使用的实现**：
 * `@mkkellogg/gaussian-splats-3d`（MIT，three.js，WebGL2，支持 .ply/.splat/.ksplat），
 * 它对 3DGS PLY 的 59 维（含 SH）有完整支持。
 *
 * 本文件只做**适配**（加载、取景、尺寸、销毁），渲染全部交给该库。
 *
 * vendored 于 `/static/vendor/`（离线可用）：
 *   gaussian-splats-3d.module.js  v0.4.7  MIT
 *   three.module.js               r160
 */
import * as THREE from 'three';
import * as GaussianSplats3D from './vendor/gaussian-splats-3d.module.js';

let viewer = null;
let hostEl = null;
let stageEl = null;

function disposeViewer() {
  if (viewer) {
    try { viewer.stop(); } catch (e) { /* 忽略：某些版本没有 start 过 */ }
    try { viewer.dispose(); } catch (e) { /* 同上 */ }
    viewer = null;
  }
  // ⚠️ **只删自己的容器**。
  //
  // 曾经这里是 `hostEl.innerHTML = ''`，而 hostEl 是 `#m-canvas-wrap` ——
  // 于是把 `#m-overlay` / `#m-canvas` / `#m-image` 一并删掉了。后果：
  //   * 任何 `$('m-overlay').classList` 都变成 null 报错
  //     （用户看到的「服务端渲染失败：Cannot read properties of null (reading 'classList')」）；
  //   * glb / 普通 ply 没有画布 ⇒ 看不了；
  //   * `GsViewer` 没有画布 ⇒ 重置视角 / 切换线框 / 旋转模式全部失效。
  // 一个"清空宿主"的写法，连带毁掉三个功能 —— 所以这里改成独立子容器。
  if (stageEl && stageEl.parentNode) {
    stageEl.parentNode.removeChild(stageEl);
  }
  stageEl = null;
}

/** 在宿主容器里建（或复用）一个只属于参考渲染器的子容器。 */
function ensureStage(host) {
  let el = host.querySelector(':scope > .gs-ref-stage');
  if (!el) {
    el = document.createElement('div');
    el.className = 'gs-ref-stage';
    host.appendChild(el);
  }
  return el;
}

/**
 * 挂载并加载一个场景。
 *
 * @param {HTMLElement} host 容器（会被清空并塞入库自己创建的 canvas）
 * @param {string} url 模型地址：`.ply`（推荐，59 维含 SH）或 `.splat`
 * @param {object} info `/api/scene_info/{job}` 的结果：
 *        { center, radius, up, cameraPosition, cameraLookAt }
 * @param {(msg:string)=>void} onStatus 状态回调（显示在界面状态栏）
 */
export async function mount(host, url, info, onStatus) {
  disposeViewer();
  hostEl = host;
  stageEl = ensureStage(host);
  const say = (m) => { if (typeof onStatus === 'function') onStatus(m); };

  const center = (info && info.center) || [0, 0, 0];
  const up = (info && info.up) || [0, 1, 0];
  const camPos = (info && info.cameraPosition) || [0, 0, 8];
  const lookAt = (info && info.cameraLookAt) || center;
  say('初始化参考渲染器…');

  viewer = new GaussianSplats3D.Viewer({
    // 库自己创建 canvas 并填满 rootElement —— 这里给的是**专属子容器**，
    // 不是 `#m-canvas-wrap`，避免它清空宿主 DOM（见 disposeViewer 的说明）。
    rootElement: stageEl,
    cameraUp: up,
    initialCameraPosition: camPos,
    initialCameraLookAt: lookAt,
    useBuiltInControls: true,
    selfDrivenMode: true,
    // 让背景与查看器原有的深色一致，便于对比
    sharedMemoryForWorkers: false,
  });

  say('加载模型…');
  await viewer.addSplatScene(url, {
    showLoadingUI: true,
    splatAlphaRemovalThreshold: 5,
    progressiveLoad: true,
  });
  viewer.start();
  const n = (info && info.gaussians) || 0;
  say(`参考渲染器就绪${n ? '（' + n.toLocaleString() + ' 高斯）' : ''}`);
  return {
    renderer: 'gaussian-splats-3d v0.4.7 (three.js r160)',
    gaussians: n,
    up,
  };
}

export function resetCamera(info) {
  if (!viewer) return;
  try {
    if (info && info.cameraPosition) {
      viewer.camera.position.set(...info.cameraPosition);
      const c = (info.cameraLookAt) || [0, 0, 0];
      viewer.controls.target.set(...c);
      viewer.controls.update();
    }
  } catch (e) { /* 忽略 */ }
}

export function dispose() { disposeViewer(); }

export function available() { return !!GaussianSplats3D.Viewer; }

// 供经典脚本（app.js）使用：模块加载完成后挂到 window 上
window.GsRefViewer = { mount, dispose, resetCamera, available, THREE };
window.dispatchEvent(new Event('gs-refviewer-ready'));
