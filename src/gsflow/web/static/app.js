/* gsflow GUI —— 无框架、无 CDN 依赖，离线可用。 */
'use strict';

// 不透明度治理方案的映射表（**由后端 /api/opacity_cleanup 填充**，不在前端硬编码）。
// 这样 CLI 的 --opacity-cleanup 与 GUI 下拉框一定产出相同的参数集合。
const OPACITY_MAP = {};
async function loadOpacityMap() {
  try {
    const d = await api('/api/opacity_cleanup');
    (d.choices || []).forEach((c) => { OPACITY_MAP[c.value] = c.params; });
    console.debug('[opacity_cleanup] 已从后端载入映射：', Object.keys(OPACITY_MAP));
  } catch (e) {
    console.warn('[opacity_cleanup] 无法从后端载入映射，将只发送方案名：', e);
  }
}

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const res = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, opts || {}));
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.headers.get('content-type')?.includes('json') ? res.json() : res.text();
};

// 拍摄模式的映射表（**由后端 /api/capture_modes 填充**，不在前端硬编码）。
// 这样 CLI 的 --capture-mode 与 GUI 下拉框一定产出相同的参数集合。
const CAPTURE_MAP = {};
async function loadCaptureMap() {
  try {
    const d = await api('/api/capture_modes');
    (d.choices || []).forEach((c) => { CAPTURE_MAP[c.value] = c.params; });
    console.debug('[capture_mode] 已从后端载入映射：', Object.keys(CAPTURE_MAP));
  } catch (e) {
    console.warn('[capture_mode] 无法从后端载入映射，将只发送模式名：', e);
  }
}

let STAGES = [];
let CONFIG = {};
let CONFIG_PATH = '';
let CURRENT_RUN = null;
let CURRENT_CURSOR = 0;
let EVT_SOURCE = null;

/* ------------------------------------------------------------------ tabs */
document.querySelectorAll('.tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    $('panel-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'jobs') refreshJobs();
    if (btn.dataset.tab === 'doctor') runDoctor();
    if (btn.dataset.tab === 'models') initModelsPage();
    if (btn.dataset.tab === 'mesh') initMeshPage();
  });
});

/* ------------------------------------------------------------------ 健康 */
async function health() {
  try {
    const h = await api('/api/health');
    $('health').textContent = '就绪 · ' + h.workspace;
    $('health').classList.add('ok');
  } catch (e) {
    $('health').textContent = '服务异常：' + e.message;
  }
}

/* ------------------------------------------------------------------ 阶段 */
async function loadStages() {
  STAGES = await api('/api/stages');
  const box = $('stage-list');
  box.innerHTML = '';
  STAGES.forEach((s, i) => {
    const el = document.createElement('label');
    el.className = 'stage-item on';
    el.innerHTML = `<input type="checkbox" checked data-stage="${s.name}">
      <span><span class="t">${s.title}</span><br><span class="d">${s.description}</span></span>`;
    el.querySelector('input').addEventListener('change', (e) => {
      el.classList.toggle('on', e.target.checked);
      updateCliPreview();
    });
    box.appendChild(el);
  });
  updateCliPreview();
}

const checkedStages = () => Array.from(document.querySelectorAll('#stage-list input:checked')).map((i) => i.dataset.stage);

/* --------------------------------------------------------------- 参数收集 */
/**
 * 解析「高级参数」文本框：每行 `段.键=值`（`#` 开头为注释）。
 *
 * 为什么要这个：GUI 的参数控件是**手工 HTML**（22 个），所以任何没写控件的配置键
 * 在界面上都改不到 —— 实测有 **9 个**已实现且有用的训练旋钮属于这种
 * （`ssim_weight` / `eval_interval` / `white_background` / `pause_refine_after_reset`
 *  / `min_imgs_for_holdout` / `eval_quick_views` / `log_every` / `flip_camera` /
 *  `reset_opacity`）。用户明确提过"参数同步"的问题。
 * 有了这个框，**任何**配置键都能从界面设置，而且以后新增参数不必再改前端。
 */
function parseAdvancedParams(text) {
  const out = {};
  const bad = [];
  String(text || '').split(/\r?\n/).forEach((raw) => {
    const line = raw.trim();
    if (!line || line.startsWith('#')) return;
    const i = line.indexOf('=');
    if (i <= 0) { bad.push(line); return; }
    const k = line.slice(0, i).trim();
    const v = line.slice(i + 1).trim();
    if (!k.includes('.')) { bad.push(line); return; }
    out[k] = v;
  });
  if (bad.length) {
    console.warn('[参数] 以下行格式不对（应为 段.键=值）：', bad);
  }
  return out;
}

function collectParams() {
  const p = {};
  const put = (k, v) => { if (v !== '' && v !== null && v !== undefined) p[k] = v; };
  put('mask.backend', $('p-mask-backend').value);
  put('mask.mode', $('p-mask-mode').value);
  // 拍摄模式：映射**从后端取**（/api/capture_modes），不在 JS 里重写 ——
  // 与 --capture-mode 共用 `gsdata/capture_mode_params.py` 一份定义。
  // 注意：显式选择了模式时，模式带来的键（如 turntable 的 mask.mode=keep）
  // **覆盖**上面的手动选择，避免"两个地方都能设、结果取决于顺序"的混乱。
  const capSel = $('p-capture-mode');
  if (capSel) {
    const mode = capSel.value || 'auto';
    put('capture.mode', mode);
    const mapped = CAPTURE_MAP[mode];
    if (mapped) Object.entries(mapped).forEach(([k, v]) => { p[k] = v; });
  }
  put('mask.prompt', $('p-mask-prompt').value.trim());
  put('frames.fps', $('p-frames-fps').value);
  put('frames.long_side', $('p-frames-long').value);
  put('colmap.matcher', $('p-colmap-matcher').value);
  put('train.backend', $('p-train-backend').value);
  put('train.iterations', $('p-train-iters').value);
  put('train.resolution', $('p-train-res').value);
  put('train.cap_max', $('p-train-cap').value);
  // 实测校准过的两个训练参数（见 docs/03 G13/G17）
  put('train.densify_until', $('p-train-densify').value);
  put('train.mask_loss_until', $('p-train-maskloss').value);
  // 网格导出
  put('export.formats', $('p-export-formats').value.trim());
  put('export.mesh_method', $('p-mesh-method').value);
  put('export.mesh_poisson_depth', $('p-mesh-depth').value);
  put('export.mesh_target_triangles', $('p-mesh-tris').value);
  put('export.mesh_smooth_iters', $('p-mesh-smooth').value);
  put('export.mesh_width', $('p-mesh-width').value);
  put('export.mesh_max_views', $('p-mesh-views').value);
  put('export.mesh_density_quantile', $('p-mesh-densq').value);
  put('export.mesh_texture', $('p-mesh-texture').value);
  put('export.mesh_texture_size', $('p-mesh-texsize').value);
  // 不透明度治理方案：映射**从后端取**（/api/opacity_cleanup），不在 JS 里重写。
  // 原因：CLI 的 --opacity-cleanup 也走同一份 `trainers/opacity_cleanup.py`，
  // 两边共用一处定义才不会漂移 —— 本项目因"同一决定写两遍"踩过多次坑
  // （最严重的一次是 22 个脚本都用 sh_degree=0 渲染，白丢 1.91 dB）。
  // 若接口还没返回（首屏），退回用 choice 本身作为标记，不猜具体参数。
  const ocSel = $('p-train-opacityclean');
  if (ocSel) {
    const choice = ocSel.value || 'none';
    const param = $('p-train-opacityparam');
    const num = parseFloat(param && param.value ? param.value : '');
    put('train.opacity_cleanup', choice);
    const mapped = OPACITY_MAP[choice];
    if (mapped) {
      Object.entries(mapped).forEach(([k, v]) => {
        // 只有"数值可调"的那些键才用界面上的数字覆盖
        if (/quantile$|^train\.opacity_decay$/.test(k) && !Number.isNaN(num)) {
          p[k] = String(num);
        } else {
          p[k] = v;
        }
      });
    }
  }
  // 高级参数：覆盖上面的同名键（放在最后 ⇒ 以用户显式输入为准）
  const adv = $('p-advanced') ? parseAdvancedParams($('p-advanced').value) : {};
  Object.assign(p, adv);
  return p;
}

function collectRunBody(dryRun) {
  const input = $('in-path').value.trim();
  return {
    input_path: input,
    images_dir: '',
    job_id: $('in-jobid').value.trim(),
    title: $('in-title').value.trim() || '3DGS 重建',
    stages: checkedStages(),
    params: collectParams(),
    force_stages: $('in-force').value.split(',').map((s) => s.trim()).filter(Boolean),
    halt_on_error: $('in-halt').checked,
    dry_run: !!dryRun,
  };
}

function updateCliPreview() {
  const b = collectRunBody(false);
  const parts = ['gsflow run'];
  if (b.images_dir) parts.push(`--images "${b.images_dir}"`);
  else if (b.input_path) parts.push(`--images "${b.input_path}"`);
  if (b.job_id) parts.push(`--job-id ${b.job_id}`);
  if (b.title !== '3DGS 重建') parts.push(`--title "${b.title}"`);
  if (b.stages.length) parts.push('--stages ' + b.stages.join(','));
  if (b.force_stages.length) parts.push('--force ' + b.force_stages.join(','));
  Object.entries(b.params).forEach(([k, v]) => parts.push(`--set ${k}=${v}`));
  if (b.dry_run) parts.push('--dry-run');
  $('cli-preview').textContent = parts.join(' \\\n  ');
}

/* ------------------------------------------------------------------ 提交 */
async function submitRun(dryRun) {
  const body = collectRunBody(dryRun);
  if (!body.input_path && !body.job_id) { alert('请填写输入路径，或填写已有作业 ID 做续跑/单阶段重跑'); return; }
  try {
    const res = await api('/api/run', { method: 'POST', body: JSON.stringify(body) });
    CURRENT_RUN = res.run_id;
    CURRENT_CURSOR = 0;
    $('log-view').textContent = '';
    $('live-title').textContent = res.job_id || res.run_id;
    $('btn-cancel').disabled = false;
    document.querySelector('.tab[data-tab="jobs"]').click();
    attachStream(res.run_id);
  } catch (e) {
    alert('提交失败：' + e.message);
  }
}

function attachStream(runId) {
  if (EVT_SOURCE) { EVT_SOURCE.close(); EVT_SOURCE = null; }
  const es = new EventSource(`/api/run/${runId}/stream?since=${CURRENT_CURSOR}`);
  EVT_SOURCE = es;
  es.onmessage = (ev) => {
    let e; try { e = JSON.parse(ev.data); } catch (err) { return; }
    CURRENT_CURSOR = Math.max(CURRENT_CURSOR, e.seq || 0);
    appendLog(e);
  };
  es.addEventListener('end', (ev) => {
    try {
      const d = JSON.parse(ev.data);
      appendLog({ level: d.status === 'done' ? 'success' : 'error', message: '作业结束：' + d.status, ts: Date.now() / 1000 });
    } catch (err) { /* ignore */ }
    es.close(); EVT_SOURCE = null;
    $('btn-cancel').disabled = true;
    refreshJobs();
  });
  es.onerror = () => { es.close(); EVT_SOURCE = null; };
}

function appendLog(e) {
  const box = $('log-view');
  const t = new Date((e.ts || Date.now() / 1000) * 1000).toTimeString().slice(0, 8);
  const tag = e.stage ? `[${e.stage}]` : '';
  const prog = (e.progress !== null && e.progress !== undefined) ? ` ${(e.progress * 100).toFixed(1)}%` : '';
  const line = document.createElement('div');
  line.className = 'lvl-' + (e.level || 'info');
  line.textContent = `${t} ${tag}${prog} ${e.message}`;
  box.appendChild(line);
  if (e.progress !== null && e.progress !== undefined) $('progress-bar').style.width = (e.progress * 100) + '%';
  box.scrollTop = box.scrollHeight;
  while (box.childElementCount > 1500) box.removeChild(box.firstChild);
}

/* ------------------------------------------------------------------ 作业 */
async function refreshJobs() {
  try {
    const d = await api('/api/jobs');
    const runBox = $('running-list');
    const hisBox = $('history-list');
    runBox.innerHTML = '';
    hisBox.innerHTML = '';
    if (!d.running.length) runBox.innerHTML = '<p class="hint">暂无运行中的作业</p>';
    d.running.forEach((r) => {
      const el = document.createElement('div');
      el.className = 'item';
      el.innerHTML = `<div class="top"><span class="jid">${r.job_id || r.run_id}</span>
        <span class="pill ${r.status}">${r.status}</span></div>
        <div class="meta">${r.title || ''} ${r.error ? '· ' + r.error : ''}</div>`;
      el.onclick = () => { CURRENT_RUN = r.run_id; CURRENT_CURSOR = 0; $('log-view').textContent = ''; attachStream(r.run_id); };
      runBox.appendChild(el);
    });
    if (!d.history.length) hisBox.innerHTML = '<p class="hint">历史为空</p>';
    d.history.slice(0, 40).forEach((h) => {
      const stages = Object.entries(h.stages || {}).map(([k, v]) => `${k}:${v.status}`).join('  ');
      const el = document.createElement('div');
      el.className = 'item';
      el.innerHTML = `<div class="top"><span class="jid">${h.job_id}</span>
        <span class="pill ${h.status}">${h.status || '?'}</span></div>
        <div class="meta">${h.title || ''}</div><div class="meta">${stages}</div>`;
      el.onclick = () => showJob(h.job_id);
      hisBox.appendChild(el);
    });
  } catch (e) { /* ignore */ }
}

async function showJob(jobId) {
  try {
    const d = await api('/api/jobs/' + jobId);
    $('live-title').textContent = jobId;
    $('log-view').textContent = '';
    const logs = await api('/api/logs/' + jobId + '?tail=400');
    (logs.events || []).forEach((line) => appendLog(JSON.parse(line)));
    const arts = [];
    Object.entries(d.stages || {}).forEach(([k, v]) => (v.artifacts || []).forEach((a) => arts.push(a)));
    if (arts.length) {
      const el = document.createElement('div');
      el.className = 'meta';
      // ⚠️ **不要再自己拼 URL**。服务端 `/api/jobs/{id}` 已把产物归一化成
      //    {path, rel, name, url}（api 的 _normalize_artifacts）。
      //    这里踩过两个叠加的坑，导致产物链接**全部 404**：
      //      1) 服务端原先返回**绝对路径**（F:\3dgs-and-ai\workspace\...），
      //         绝对路径在工作区之外，URL 里无论怎么编码都取不到；
      //      2) 前端又把反斜杠替换写成了 `/\\\\/g`（匹配**两个**反斜杠），
      //         而 Windows 路径是单个反斜杠 ⇒ 替换不触发 ⇒ `\` 被编成 `%5C`。
      //    现在服务端给 url，前端只负责用；只有老数据（字符串）才走兜底。
      // ⚠️ 产物里**既有文件也有目录**（如 `01_images_raw`、`02_masks`、`03_colmap/sparse/0`）。
      //    端点 `/api/artifacts/...` 对目录**正确地**返回 404（它只服务文件），
      //    所以目录**不能再渲染成可点击链接** —— 否则用户一点就是
      //    "Failed to load resource: 404"。目录用无链接的样式标出即可。
      el.innerHTML = '产物：' + arts.slice(0, 12).map((a) => {
        const label = (a && (a.rel || a.name)) ? (a.rel || a.name) : String(a);
        const isDir = /\/$/.test(label) || (a && a.is_dir);
        if (isDir) return `<span class="dim" title="目录（不提供直接下载）">${label}/</span>`;
        const href = (a && a.url) ? a.url
          : `/api/artifacts/${encodeURIComponent(jobId)}/` +
            encodeURIComponent(String(a).replace(/\\/g, '/'));
        return `<a href="${href}" target="_blank">${label}</a>`;
      }).join(' · ');
      $('log-view').appendChild(el);
    }
  } catch (e) { alert(e.message); }
}

/* ------------------------------------------------------------------ doctor */
async function runDoctor() {
  $('doctor-view').textContent = '检测中…';
  try {
    const rep = await api('/api/doctor?save=true');
    const plan = $('doctor-plan');
    plan.innerHTML = '';
    (rep.plan || []).forEach((p) => {
      const el = document.createElement('div');
      el.className = 'p ' + p.severity;
      el.innerHTML = `<b>${p.item}</b><div class="a">${p.action}</div>`;
      plan.appendChild(el);
    });
    $('doctor-view').textContent = renderDoctorText(rep);
  } catch (e) {
    $('doctor-view').textContent = '检测失败：' + e.message;
  }
}

function renderDoctorText(r) {
  const L = [];
  L.push(`系统      : ${r.platform.system} ${r.platform.release} (${r.platform.version})`);
  L.push(`Python    : ${r.venv.python}  venv=${r.venv.in_venv}`);
  if (r.memory.total_gb) L.push(`内存      : 总 ${r.memory.total_gb} GB / 可用 ${r.memory.available_gb} GB`);
  (r.gpu.gpus || []).forEach((g) => L.push(`GPU       : ${g.name} ${Math.round(g.vram_total_mb)}MB 驱动 ${g.driver} 算力 ${g.compute_cap}`));
  L.push(`CUDA      : nvcc=${r.cuda.nvcc ? '有' : '无'} ${(r.cuda.toolkit_versions || []).join(',')}`);
  if (r.torch.installed) L.push(`torch     : ${r.torch.version} cuda=${r.torch.cuda_build} 可用=${r.torch.cuda_available} ${r.torch.compute_capability || ''} 内核=${r.torch.kernel_support}`);
  else L.push('torch     : 未安装');
  L.push('磁盘      : ' + r.disks.map((d) => `${d.drive} 余${d.free_gb}G/${d.total_gb}G`).join('  '));
  L.push('');
  L.push('外部工具：');
  r.tools.forEach((t) => L.push(`  ${t.found ? '[OK]  ' : '[MISS]'} ${t.name.padEnd(8)} ${t.version || t.advice || ''}`));
  L.push('');
  L.push('Python 依赖：');
  r.packages.forEach((p) => L.push(`  ${p.installed ? '[OK]  ' : '[MISS]'} ${p.module.padEnd(14)} ${p.version || ''}`));
  L.push('');
  L.push('网络：');
  Object.entries(r.network).forEach(([k, v]) => { if (k.endsWith('_reachable')) L.push(`  ${v ? '[OK]  ' : '[FAIL]'} ${k.replace('_reachable', '').replace(/_/g, '.')}`); });
  return L.join('\n');
}

/* ------------------------------------------------------------------ 设置 */
const CONFIG_KEYS = [
  ['project.workspace_dir', '作业根目录'],
  ['frames.mode', '抽帧模式(fps/count)'],
  ['frames.fps', '抽帧 fps'],
  ['frames.long_side', '图片长边'],
  ['frames.blur_threshold', '模糊剔除阈值'],
  ['mask.backend', '分割后端'],
  ['mask.mode', '清理模式'],
  ['mask.dilate_px', '掩码膨胀像素'],
  ['mask.feather_px', '掩码羽化像素'],
  ['mask.keep_largest', '只保留最大前景'],
  ['colmap.cameras', 'COLMAP 相机模型'],
  ['colmap.max_image_size', '特征提取图像上限'],
  ['colmap.max_num_features', '最大特征数'],
  ['colmap.matcher', '匹配模式'],
  ['colmap.use_masks', '使用掩码'],
  ['colmap.quality_min_registered_ratio', '注册率下限'],
  ['train.backend', '训练后端'],
  ['train.iterations', '迭代次数'],
  ['train.resolution', '训练分辨率'],
  ['train.cap_max', '高斯上限'],
  ['train.densify_until', '密度控制窗口上界'],
  ['train.densify_from', '密度控制起始步'],
  ['train.mask_loss_until', '掩码内损失占比'],
  ['train.eval', '启用评估'],
  ['export.formats', '导出格式'],
  ['export.glb_max_points', 'GLB 点数上限'],
  ['export.mesh_method', '网格提取方法'],
  ['export.mesh_poisson_depth', '泊松八叉树深度'],
  ['export.mesh_target_triangles', '网格目标面数'],
  ['export.mesh_smooth_iters', '网格平滑迭代'],
  ['export.mesh_width', '深度渲染分辨率'],
  ['export.mesh_max_views', '融合视角上限'],
  ['export.mesh_density_quantile', '网格密度裁剪分位'],
  ['export.mesh_voxel_div', '网格体素细分'],
  ['export.mesh_min_alpha', '网格最小不透明度'],
  ['export.mesh_texture', '纹理烘焙开关'],
  ['export.mesh_texture_size', '贴图分辨率'],
  ['runtime.max_parallel_jobs', '并发作业数'],
];

const TOOL_KEYS = [
  ['tools.ffmpeg', 'ffmpeg.exe 路径'],
  ['tools.ffprobe', 'ffprobe.exe 路径'],
  ['tools.colmap', 'colmap.exe 路径'],
  ['tools.magick', 'magick.exe 路径'],
];

function getPath(obj, path) {
  return path.split('.').reduce((acc, k) => (acc == null ? undefined : acc[k]), obj);
}

function buildEditor(containerId, keys) {
  const box = $(containerId);
  box.innerHTML = '';
  keys.forEach(([path]) => {
    const val = getPath(CONFIG, path);
    if (val === undefined) return;
    const row = document.createElement('div');
    row.className = 'entry';
    const input = document.createElement('input');
    input.dataset.path = path;
    input.value = Array.isArray(val) ? val.join(',') : (val === null ? '' : val);
    row.innerHTML = `<span>${path}</span>`;
    row.appendChild(input);
    box.appendChild(row);
  });
}

async function loadConfig() {
  const d = await api('/api/config');
  CONFIG = d.config; CONFIG_PATH = d.config_path;
  buildEditor('config-editor', CONFIG_KEYS);
  buildEditor('tools-editor', TOOL_KEYS);
}

async function saveConfig() {
  const set = {};
  document.querySelectorAll('#config-editor input, #tools-editor input').forEach((i) => {
    set[i.dataset.path] = i.value;
  });
  await api('/api/config', { method: 'POST', body: JSON.stringify({ set }) });
  await loadConfig();
  alert('配置已保存到 ' + CONFIG_PATH);
}

async function loadCreds() {
  try {
    const d = await api('/api/credentials');
    const m = d.masked;
    Object.keys(m).forEach((k) => { const el = $('c-' + k); if (el) el.placeholder = m[k] || el.placeholder; });
  } catch (e) { /* ignore */ }
}

async function saveCreds() {
  const body = {};
  ['hf_token', 'hf_endpoint', 'sam3_api_key', 'openai_api_key', 'http_proxy', 'https_proxy'].forEach((k) => {
    const el = $('c-' + k);
    if (el && el.value.trim()) body[k] = el.value.trim();
  });
  await api('/api/credentials', { method: 'POST', body: JSON.stringify(body) });
  ['hf_token', 'sam3_api_key', 'openai_api_key'].forEach((k) => { const el = $('c-' + k); if (el) el.value = ''; });
  alert('凭据已写入 config/credentials.json（只保存在本机）');
  loadCreds();
}

async function testNet() {
  $('net-result').textContent = '检测中…';
  const rep = await api('/api/doctor');
  const n = rep.network || {};
  $('net-result').textContent = Object.entries(n).filter(([k]) => k.endsWith('_reachable'))
    .map(([k, v]) => `${k.replace('_reachable', '').replace(/_/g, '.')}: ${v ? '可达' : '不可达'}`).join('   ');
}

/* ------------------------------------------------------------------ 绑定 */
$('btn-run').onclick = () => submitRun(false);
$('btn-dry').onclick = () => submitRun(true);
$('btn-refresh').onclick = refreshJobs;
$('btn-doctor').onclick = runDoctor;
$('btn-cancel').onclick = async () => { if (CURRENT_RUN) { await api(`/api/run/${CURRENT_RUN}/cancel`, { method: 'POST' }); } };
$('btn-save-creds').onclick = saveCreds;
$('btn-test-net').onclick = testNet;
$('btn-save-config').onclick = saveConfig;
$('btn-reload-config').onclick = loadConfig;
$('btn-reset-config').onclick = async () => { await api('/api/config/reset', { method: 'POST' }); loadConfig(); };
['in-path', 'in-title', 'in-jobid', 'in-force', 'p-mask-backend', 'p-mask-mode', 'p-mask-prompt',
  'p-capture-mode',
  'p-frames-fps', 'p-frames-long', 'p-colmap-matcher', 'p-train-backend', 'p-train-iters',
  'p-train-res', 'p-train-cap', 'p-train-densify', 'p-train-maskloss',
  'p-train-opacityclean', 'p-train-opacityparam',
  'p-export-formats', 'p-mesh-method', 'p-mesh-depth', 'p-mesh-tris', 'p-mesh-smooth',
  'p-mesh-width', 'p-mesh-views', 'p-mesh-densq', 'p-mesh-texture', 'p-mesh-texsize',
  'p-advanced',
].forEach((id) => { const el = $(id); if (el) el.addEventListener('input', updateCliPreview); });
// <select> 的变化用 change 事件（input 在部分浏览器上对 select 不触发）
['p-train-opacityclean'].forEach((id) => {
  const el = $(id); if (el) el.addEventListener('change', updateCliPreview);
});

/* ------------------------------------------------------ 模型查看（独立功能页） */
/* ============================================================ 网格导出页 */
/**
 * 对**已训练**的作业单独重导出网格。
 *
 * 为什么单独做一页：网格提取的调参是**反复试**的过程（换方法、调泊松深度、
 * 调目标面数），而重跑一次训练要十几分钟。这一页只触发 `export` 阶段，
 * 并显示等价的 CLI 命令，保证 GUI 与 CLI 行为一致（架构红线）。
 */
let MESH_READY = false;

function initMeshPage() {
  if (!MESH_READY) {
    MESH_READY = true;
    const reload = $('btn-me-jobs');
    if (reload) reload.onclick = loadMeshJobs;
    const dry = $('btn-me-dry');
    if (dry) dry.onclick = () => runMeshExport(true);
    const run = $('btn-me-run');
    if (run) run.onclick = () => runMeshExport(false);
  }
  loadMeshJobs();
}

function collectMeshParams() {
  const p = {};
  const put = (k, v) => { if (v !== '' && v !== null && v !== undefined) p[k] = v; };
  put('export.formats', $('me-formats').value.trim());
  put('export.mesh_method', $('me-method').value);
  put('export.mesh_poisson_depth', $('me-depth').value);
  put('export.mesh_target_triangles', $('me-tris').value);
  put('export.mesh_width', $('me-width').value);
  put('export.mesh_max_views', $('me-views').value);
  put('export.mesh_smooth_iters', $('me-smooth').value);
  put('export.mesh_texture', $('me-texture').value);
  put('export.mesh_texture_size', $('me-texsize').value);
  // 只有 2dgs 模式读这几项，其它模式传了也无害
  put('export.mesh_consistency_min_views', $('me-minviews').value);
  put('export.mesh_consistency_voxel_scale', $('me-voxscale').value);
  put('export.mesh_consistency_normal_dot', $('me-nrmdot').value);
  return p;
}

async function loadMeshJobs() {
  const sel = $('me-job');
  if (!sel) return;
  try {
    const j = await api('/api/jobs');
    const all = [].concat(j.running || [], j.history || []);
    const prev = sel.value;
    sel.innerHTML = '';
    if (!all.length) {
      sel.innerHTML = '<option value="">（没有作业）</option>';
      return;
    }
    all.forEach((x) => {
      const o = document.createElement('option');
      o.value = x.job_id;
      o.textContent = x.job_id + (x.status ? `（${x.status}）` : '');
      sel.appendChild(o);
    });
    if (prev && all.some((x) => x.job_id === prev)) sel.value = prev;
  } catch (e) {
    sel.innerHTML = '<option value="">载入失败：' + e.message + '</option>';
  }
}

async function runMeshExport(dryRun) {
  const job = $('me-job').value;
  const st = $('me-status');
  const note = $('me-note');
  if (!job) { if (st) st.textContent = '请先选择一个作业。'; return; }
  const method = $('me-method').value;

  // 参数预检：把已知"必然失败/极慢"的组合提前提示。阈值都来自实测（见 docs/03 G46/G50）。
  const depth = parseInt($('me-depth').value, 10);
  const tris = parseInt($('me-tris').value, 10);
  const warn = [];
  if (method === 'poisson' && depth >= 10) {
    warn.push('泊松深度 ≥10 在本机（15GB 内存）实测不可用：2771 万面会让后处理'
      + '超过 27 分钟。建议保持 9；要更细请提高「深度图分辨率」或减少参与视角。');
  }
  if (method === '2dgs' && parseInt($('me-minviews').value, 10) > 4) {
    warn.push('最小视角数 >4 会删掉大量真实表面'
      + '（80 视角下每个表面点通常只被 2~3 个视角看到）。');
  }
  if ($('me-texture').value === 'true' && tris > 30000) {
    warn.push('目标面数 >3 万时烘焙贴图很可能超时（xatlas 在 8 万面上要 13 分钟以上）。');
  }
  if (note) note.innerHTML = warn.length ? ('⚠ ' + warn.join('<br>⚠ ')) : '';

  const body = {
    job_id: job,
    title: '网格导出 · ' + method,
    stages: ['export'],
    force_stages: ['export'],
    params: collectMeshParams(),
    dry_run: !!dryRun,
  };
  if (st) {
    st.textContent = dryRun ? '正在生成命令预览…'
      : '已提交，正在导出（可在「作业与进度」看日志）…';
  }
  try {
    const r = await api('/api/run', { method: 'POST', body: JSON.stringify(body) });
    const cli = $('me-cli');
    if (cli) cli.textContent = r.cli_equivalent || '';
    if (st) {
      st.textContent = dryRun
        ? '以上是本次导出等价的 CLI 命令（GUI 与 CLI 走同一套流程逻辑）。'
        : `已提交：run_id=${r.run_id || '-'} job=${r.job_id || job}。`
          + '导出完成后到「模型查看」页刷新即可看到新网格。';
    }
  } catch (e) {
    if (st) st.textContent = '失败：' + e.message;
  }
}

/* 三种模式对应用户要求的三种查看方式：
 *   points   点云（model.glb / model.ply）      —— 3DGS 原始形态
 *   mesh     三角网格 + 顶点色（model_mesh.glb）—— 看几何
 *   textured 带贴图网格（model_mesh_textured.glb）—— 最终"实感"产物
 */
const MODE_DEF = [
  // 服务端渲染预览放**第一位**：它不依赖 WebGL。
  //
  // 为什么：WebGL 泼溅在用户环境里两次打不开（黑屏），而 `preview_*.png` 是
  // **训练用的同一个 gsplat 渲染器**在服务端出的图（左=渲染 右=真值）。
  // 无论浏览器什么状况，用户都能在 GUI 里看到模型的真实外观 ——
  // 这也是"看到的效果应与预览图一致"这条要求的可操作形式。
  { key: 'preview',  title: '服务端预览',  desc: 'gsplat 渲染（不依赖 WebGL，最可靠）',
    prefers: ['image'] },
  // 高斯泼溅：读 3DGS PLY 的**全部 59 维**（位置3+尺度3+旋转4+不透明度1+SH48），
  // 专用泼溅渲染器画"朝向各异的椭球 + alpha 合成 + 视角相关 SH 着色"。
  { key: 'splat',    title: '高斯泼溅',    desc: '真实高斯（59 维，含 SH 视角着色）',
    prefers: ['gauss3d', 'splat'] },
  { key: 'points',   title: '点云',        desc: '3DGS 原始形态，最忠实',
    prefers: ['glb_points', 'ply', 'spz'] },
  { key: 'mesh',     title: '网格',        desc: '三角面 + 顶点色，看几何',
    prefers: ['glb_mesh', 'obj_mesh', 'ply_mesh'] },
  { key: 'textured', title: '带贴图网格',  desc: '最终"实感"产物',
    prefers: ['glb_textured'] },
];
let MODELS = null;          // 当前作业的 /api/models 结果
let ACTIVE_MODE = 'preview'; // 默认显示最可靠的视图（没有则自动回退）
let VIEWER_READY = false;

/* ============================================================================
 * ViewerHub：**同一时刻只有一个活动渲染器**，按钮只作用于它
 * ============================================================================
 *
 * ## 为什么要这层（2026-10-25，用户反馈"按钮交互会冲突"）
 *
 * 页面里存在两套渲染器：
 *   * `GsRefViewer`（refviewer.js，社区参考库 gaussian-splats-3d）—— 负责 3DGS 的 .ply/.splat；
 *   * `GsViewer`（viewer.js，自研 WebGL1）—— 负责 glb / 普通 ply / 网格。
 *
 * 但按钮（重置视角 / 切换线框 / 旋转模式）以前**无条件打到 `GsViewer`**：
 *   * 显示 3DGS 时 `GsViewer` 并没有画布在显示 ⇒ 点"重置"看着像没反应；
 *   * 更糟的是它可能作用在**隐藏的 canvas** 上，而屏幕上是参考渲染器 —— 即"命令打到另一个渲染器"。
 *
 * 这层把"活动渲染器 + 能力"显式化：
 *   * `set(kind)`      —— 切换活动渲染器，并刷新按钮可用性；
 *   * `supports(cap)`  —— 当前渲染器是否支持某能力；
 *   * `reset/wire/rot` —— 路由到当前渲染器；不支持就**什么都不做且按钮已禁用**。
 */
const ViewerHub = (() => {
  let active = null;           // 'ref' | 'gs' | null
  let refInfo = null;          // 参考渲染器的取景信息（resetCamera 需要）
  const CAPS = {
    ref: { reset: true, wire: false, rotmode: false },
    gs: { reset: true, wire: true, rotmode: true },
  };
  const WHY = {
    wire: '线框只对自研 WebGL 查看器（glb / 网格）有效；当前是 3DGS 参考渲染器',
    rotmode: '旋转模式只对自研 WebGL 查看器有效；当前是 3DGS 参考渲染器',
  };

  function supports(cap) {
    return !!(active && CAPS[active] && CAPS[active][cap]);
  }

  function syncButtons() {
    const pairs = [
      ['btn-m-wire', 'wire', '切换线框'],
      ['btn-m-rotmode', 'rotmode', null],
      ['btn-m-reset', 'reset', null],
    ];
    for (const [id, cap, label] of pairs) {
      const el = document.getElementById(id);
      if (!el) continue;
      const ok = supports(cap);
      el.disabled = !ok;
      el.style.opacity = ok ? '' : '0.45';
      el.style.cursor = ok ? '' : 'not-allowed';
      if (ok) {
        if (el.dataset.hubWhy) { delete el.dataset.hubWhy; el.removeAttribute('title'); }
        if (id === 'btn-m-rotmode') {
          try {
            const m = window.GsViewer && window.GsViewer.getRotMode
              ? window.GsViewer.getRotMode() : 'free';
            el.textContent = m === 'free' ? '旋转模式：自由翻转' : '旋转模式：到顶底停住';
          } catch (e) { /* 忽略 */ }
        } else if (label) {
          el.textContent = label;
        }
      } else {
        el.dataset.hubWhy = WHY[cap] || '当前查看器不支持该操作';
        el.title = el.dataset.hubWhy;
      }
    }
  }

  function set(kind) {
    active = (kind === 'ref' || kind === 'gs') ? kind : null;
    // 指针事件隔离：只有**活动**渲染器的画布接收鼠标/触摸。
    // 3DGS 时仅把我们的 canvas 设 display:none 还不够（某些布局下仍可命中），
    // 这里再显式关掉 pointer-events，避免一次拖拽同时驱动两套控制器 ——
    // 这正是用户说的"与 webgl 的交互没分开"。
    try {
      const cv = document.getElementById('m-canvas');
      if (cv) cv.style.pointerEvents = (active === 'gs') ? 'auto' : 'none';
      const stage = document.querySelector('#m-canvas-wrap > .gs-ref-stage');
      if (stage) stage.style.pointerEvents = (active === 'ref') ? 'auto' : 'none';
    } catch (e) { /* 忽略 */ }
    syncButtons();
    return active;
  }

  function reset() {
    if (active === 'ref' && window.GsRefViewer && window.GsRefViewer.resetCamera) {
      window.GsRefViewer.resetCamera(refInfo || {});
      return true;
    }
    if (active === 'gs' && window.GsViewer && window.GsViewer.reset) {
      window.GsViewer.reset();
      return true;
    }
    return false;
  }

  function toggleWire() {
    if (!supports('wire')) return null;
    return window.GsViewer.toggleWire();
  }

  function setRotMode(mode) {
    if (!supports('rotmode')) return null;
    window.GsViewer.setRotMode(mode);
    return window.GsViewer.getRotMode();
  }

  function rotMode() {
    if (!supports('rotmode')) return null;
    return window.GsViewer.getRotMode();
  }

  return {
    set, supports, reset, toggleWire, setRotMode, rotMode, syncButtons,
    setRefInfo(info) { refInfo = info || null; },
    get active() { return active; },
  };
})();

/**
 * 懒创建 WebGL1 查看器（只服务 glb / 普通 ply）。
 * 高斯与预览模式不碰它 —— 见 initModelsPage 里的说明。
 *
 * ⚠️ 初始化前**必须先把 canvas 显示出来**：`display:none`（或 0×0）的 canvas
 * 拿不到有效尺寸，WebGL 视口会是 0，表现为"模型看不见、按钮点了没反应"。
 */
function ensureViewer() {
  if (!window.GsViewer) return false;
  const cv = $('m-canvas');
  if (cv) cv.style.display = 'block';
  const im = $('m-image');
  if (im) { im.style.display = 'none'; }
  // 参考渲染器的层要撤掉，否则两块画面叠在一起
  if (window.GsRefViewer && window.GsRefViewer.dispose) {
    try { window.GsRefViewer.dispose(); } catch (e) { /* 忽略 */ }
  }
  if (VIEWER_READY) return true;
  try {
    window.GsViewer.init(cv, (s) => { $('m-status').textContent = s || ''; });
    VIEWER_READY = true;
    // 自研 WebGL 查看器就绪 ⇒ 它是当前活动渲染器（按钮路由到它）
    try { ViewerHub.set('gs'); } catch (e) { /* 忽略 */ }
    return true;
  } catch (e) {
    const ov = $('m-overlay');
    if (ov) {
      ov.classList.remove('hide');
      ov.textContent = '查看器初始化失败：' + e.message;
    }
    return false;
  }
}

function initModelsPage() {
  // 确认脚本到位：`viewer.js`（glb/普通 ply 的 WebGL1 查看器）与
  // `refviewer.js`（高斯，社区参考实现）各管一半，缺哪个都会表现成"黑屏"，
  // 而控制台里往往只剩一句 404，很难看出是脚本没加载。
  // `refviewer.js` 是 type=module，可能比这个经典脚本晚就绪，所以不在这里硬性要求，
  // 而是在真正要显示高斯时检查（见 loadModelFile）。
  const missing = [];
  if (!window.GsViewer) missing.push('viewer.js');
  if (missing.length) {
    $('m-overlay').classList.remove('hide');
    $('m-overlay').style.whiteSpace = 'pre-line';
    $('m-overlay').textContent =
      '前端脚本未加载：' + missing.join('、')
      + '\n\n常见原因：页面是在文件更新前打开的（浏览器缓存了旧版）。'
      + '\n请按 Ctrl+F5 强制刷新；若仍失败，打开 /api/selftest 看服务端报告。';
    $('m-status').textContent = '脚本缺失';
    return;
  }
  // ⚠️ **懒初始化**：`GsViewer` 会创建一个 WebGL1 上下文，而它只负责
  // glb / 普通 ply（网格、点云）。高斯模式用的是参考渲染器（自带 WebGL2 上下文），
  // 预览模式只显示 PNG。所以在页面加载时就 init 等于**白占一个上下文**，
  // 也让"到底哪个渲染器在画"变得不可判断（用户说的"不要硬嵌入"）。
  // 改成真正需要时再建（见 ensureViewer / loadModelFile）。
  initLocalFilePicker();
  // 服务端渲染 / 返回 WebGL：把"高质量静态图"与"可交互低延迟"两种模式接起来
  const btnSrv = $('m-srv-render');
  if (btnSrv) {
    btnSrv.onclick = () => {
      renderCurrentViewOnServer().catch((e) => {
        $('m-status').textContent = '服务端渲染失败：' + e.message;
      });
    };
  }
  const btnBack = $('m-back-webgl');
  if (btnBack) {
    btnBack.onclick = () => {
      const img = $('m-image');
      if (img) { img.style.display = 'none'; img.removeAttribute('src'); }
      const cv = $('m-canvas');
      if (cv) cv.style.display = 'block';
      if (window.GsViewer && window.GsViewer.redraw) window.GsViewer.redraw();
      $('m-overlay').classList.add('hide');
      $('m-status').textContent = 'WebGL 交互模式';
    };
  }
  loadModelJobs();
}

async function loadModelJobs() {
  try {
    const j = await api('/api/jobs');
    const all = [].concat(j.running || [], j.history || []);
    const seen = new Set();
    const opts = ['<option value="">— 选择作业 —</option>'];
    all.forEach((it) => {
      const id = it.job_id || it.id;
      if (!id || seen.has(id)) return;
      seen.add(id);
      const st = it.status || '';
      opts.push(`<option value="${id}">${id}${st ? '  (' + st + ')' : ''}</option>`);
    });
    $('m-job').innerHTML = opts.join('');
  } catch (e) {
    $('m-job').innerHTML = '<option value="">（作业列表读取失败）</option>';
  }
}

function renderModeButtons() {
  const box = $('m-modes');
  box.innerHTML = '';
  MODE_DEF.forEach((d) => {
    const n = MODELS ? (MODELS.counts?.[d.key] || 0) : 0;
    const el = document.createElement('button');
    el.className = 'mode' + (d.key === ACTIVE_MODE ? ' active' : '');
    el.innerHTML = `<div class="mt">${d.title}</div><div class="md">${d.desc}</div>`
      + `<div class="mc">${n} 个产物</div>`;
    el.disabled = n === 0;
    el.onclick = () => { ACTIVE_MODE = d.key; renderModeButtons(); renderModeFiles(); };
    box.appendChild(el);
  });
}

/**
 * 统一格式化产物大小。
 *
 * 为什么需要：后端对"训练产物 PLY"这类**不在 05_export 里**的文件一度把
 * `size_mb` 写死成 null，而模板直接拼 `${f.size_mb} MB` ⇒ 界面上显示
 * 「point_cloud.ply（完整 59 维）  null MB」，用户会以为**文件不存在**。
 * 后端已改成给真实字节数；这里再兜一层，拿不到大小就显示"—"，
 * 绝不把 `null` / `undefined` 直接印给用户。
 */
function fmtSize(f) {
  if (!f) return '—';
  const mb = (f.size_mb === null || f.size_mb === undefined) ? null : Number(f.size_mb);
  if (mb !== null && Number.isFinite(mb)) {
    if (mb >= 1024) return (mb / 1024).toFixed(2) + ' GB';
    return mb.toFixed(2) + ' MB';
  }
  if (typeof f.bytes === 'number' && f.bytes > 0) {
    const m = f.bytes / 1024 / 1024;
    return (m >= 1024 ? (m / 1024).toFixed(2) + ' GB' : m.toFixed(2) + ' MB');
  }
  return '—';
}

/**
 * 「服务端渲染当前视角」：把前端相机交给后端，用 **gsplat 官方光栅化器**出图。
 *
 * 为什么要这条路：GUI 的 WebGL 泼溅是我重新实现的渲染器（逐高斯桶排序、
 * 四边形 2σ 近似、固定低通），与 `preview_*.png` 的 gsplat 渲染**必然有差距**。
 * 任何 WebGL 复刻都消不掉这个差距；唯一能"按构造一致"的做法就是：
 * 前端只负责相机，像素由同一个渲染器算。
 *
 * 相机约定（与 `viewer.js::getCamera()` 和后端 `/api/render` 一致）：
 *   direction = [cos(pitch)·sin(yaw), sin(pitch), cos(pitch)·cos(yaw)]，单位为度。
 */
async function renderCurrentViewOnServer() {
  if (!MODELS) return;
  const job = $('m-job') && $('m-job').value;
  if (!job) { $('m-status').textContent = '先选择一个作业'; return; }
  if (!window.GsViewer || !window.GsViewer.getCamera) {
    $('m-status').textContent = '查看器未就绪，无法取当前相机';
    return;
  }
  const cam = window.GsViewer.getCamera();
  const w = Math.max(320, Math.min(1200, ($('m-canvas') ? $('m-canvas').width : 720) || 720));
  const h = Math.max(320, Math.min(1200, ($('m-canvas') ? $('m-canvas').height : 720) || 720));
  const url = `/api/render/${encodeURIComponent(job)}`
    + `?yaw=${cam.yaw.toFixed(3)}&pitch=${cam.pitch.toFixed(3)}`
    + `&dist=${cam.dist.toFixed(4)}&width=${w}&height=${h}`;
  $('m-status').textContent = '服务端渲染中…（gsplat）';
  $('m-overlay').classList.remove('hide');
  $('m-overlay').textContent = '服务端渲染中…';
  const img = $('m-image');
  const cv = $('m-canvas');
  const tmp = new Image();
  tmp.onload = () => {
    if (img) { img.style.display = 'block'; img.src = tmp.src; }
    if (cv) cv.style.display = 'none';
    // 服务端渲染/预览图模式：屏幕上没有可交互的 3D 渲染器
    try { ViewerHub.set(null); } catch (e) { /* 忽略 */ }
    $('m-overlay').classList.add('hide');
    $('m-status').textContent = `服务端渲染 yaw=${cam.yaw.toFixed(1)}° `
      + `pitch=${cam.pitch.toFixed(1)}° · gsplat 官方光栅化器`;
  };
  tmp.onerror = () => {
    $('m-overlay').classList.remove('hide');
    $('m-overlay').style.whiteSpace = 'pre-line';
    $('m-overlay').textContent = '服务端渲染失败（后端未重启或 GPU 忙）\n'
      + '地址：' + url;
    $('m-status').textContent = '服务端渲染失败';
  };
  tmp.src = url;
}

function renderModeFiles() {
  const box = $('m-list');
  box.innerHTML = '';
  if (!MODELS) { box.innerHTML = '<div class="hint">先选择一个作业。</div>'; return; }
  const files = (MODELS.modes && MODELS.modes[ACTIVE_MODE]) || [];
  if (!files.length) {
    box.innerHTML = '<div class="hint">该模式下没有<b>可查看</b>的产物。'
      + '（点云/网格对应 export.formats 里的 glb_points 与 mesh；'
      + '带贴图网格还需 export.mesh_texture=true）</div>';
  } else {
    files.forEach((f) => {
      const el = document.createElement('div');
      el.className = 'item';
      el.innerHTML = `<div class="top"><span class="jid">${f.name}</span>`
        + `<span class="muted">${fmtSize(f)}</span></div>`
        + `<div class="meta">${f.desc || f.kind}</div>`;
      el.onclick = () => loadModelFile(f);
      box.appendChild(el);
    });
    // 自动载入第一个，省一次点击
    loadModelFile(files[0]);
  }

  // 「可下载但不可查看」的产物（OBJ / SPZ / 贴图 PNG）单独列出。
  // 它们仍然有用（Blender 等外部工具），但查看器解析不了 —— 藏起来会让用户以为丢了。
  const dl = MODELS.downloadable || [];
  if (dl.length) {
    const h = document.createElement('div');
    h.className = 'hint';
    h.style.marginTop = '10px';
    h.textContent = '可下载（查看器不支持该格式，请用 Blender / MeshLab 等打开）：';
    box.appendChild(h);
    dl.forEach((f) => {
      const el = document.createElement('div');
      el.className = 'item';
      el.innerHTML = `<div class="top"><span class="jid">${f.name}</span>`
        + `<span class="muted">${fmtSize(f)}</span></div>`
        + `<div class="meta">${f.desc || f.kind}</div>`;
      el.onclick = () => window.open(f.url, '_blank');
      box.appendChild(el);
    });
  }
}

async function loadModelFile(f) {
  // ---- 高斯类产物：交给**参考实现**渲染，不再用我手写的 WebGL 渲染器 ----
  //
  // 用户的原话："光栅化你从头写没必要啊，而且也不好，你至少参考一下官方的啊，
  // 不然我这里看出来，模型差会误导我"。这是对的：自研渲染器的差距会被误读成
  // 模型质量差。这里用 `@mkkellogg/gaussian-splats-3d`（MIT，three.js，WebGL2，
  // 直接支持 3DGS PLY 的 59 维含 SH）来渲染，本文件只负责取景与状态显示。
  const isGauss = f.kind === 'gauss3d' || f.kind === 'splat'
    || /\.splat$/i.test(String(f.url || ''));
  if (isGauss) {
    const ref = window.GsRefViewer;
    if (!ref) {
      $('m-overlay').classList.remove('hide');
      $('m-overlay').style.whiteSpace = 'pre-line';
      $('m-overlay').textContent =
        '参考渲染库尚未加载完成：refviewer.js（type=module）\n'
        + '若持续如此，请确认 /static/vendor/ 下的 three.module.js 与 '
        + 'gaussian-splats-3d.module.js 存在，并按 Ctrl+F5 刷新。';
      return;
    }
    const job = $('m-job') && $('m-job').value;
    const host = $('m-canvas-wrap');
    if (!host) return;
    // 隐藏我自己的 canvas / 图片，把容器让给参考渲染器
    const cv = $('m-canvas'); if (cv) cv.style.display = 'none';
    const im = $('m-image'); if (im) { im.style.display = 'none'; im.removeAttribute('src'); }
    $('m-overlay').classList.remove('hide');
    $('m-overlay').style.whiteSpace = 'pre-line';
    $('m-overlay').textContent = '载入中…';
    $('m-status').textContent = '载入 ' + f.name + ' …';
    try {
      const info = job ? await api('/api/scene_info/' + encodeURIComponent(job)) : null;
      // 优先用 `.ply` 后缀路由：库按扩展名选解析器，且 PLY 有完整 SH
      const url = job
        ? `/api/gauss/${encodeURIComponent(job)}/point_cloud.ply`
        : f.url;
      const r = await ref.mount(host, url, info || {}, (s) => {
        $('m-status').textContent = s || '';
      });
      // 参考渲染器就绪 ⇒ 它是当前活动渲染器；线框/旋转模式对它不适用，
      // 按钮会被禁用并写明原因（避免"点了没反应"或"命令打到隐藏的另一个渲染器"）。
      ViewerHub.setRefInfo(info || {});
      ViewerHub.set('ref');
      $('m-overlay').classList.add('hide');
      $('m-status').textContent = `${f.name} · 参考渲染器 `
        + `(${r.renderer}) · ${(r.gaussians || 0).toLocaleString()} 高斯（含 SH）`;
    } catch (e) {
      $('m-overlay').classList.remove('hide');
      $('m-overlay').style.whiteSpace = 'pre-line';
      $('m-overlay').textContent = '参考渲染器加载失败：' + e.message
        + '\n地址：' + f.url;
      $('m-status').textContent = '加载失败';
    }
    return;
  }

  // ---- 图片类产物（服务端渲染预览）：不走 WebGL，直接 <img> 显示 ----
  //
  // 这条路径是 WebGL 出问题时的**兜底**：`preview_*.png` 由训练用的同一个
  // gsplat 渲染器在服务端生成，无论浏览器/驱动什么状态都能看到真实外观。
  if (f.kind === 'image' || /\.(png|jpe?g|webp)$/i.test(String(f.url || ''))) {
    // 先撤掉参考渲染器的层，否则高斯/图片两块画面会叠在一起
    if (window.GsRefViewer && window.GsRefViewer.dispose) {
      try { window.GsRefViewer.dispose(); } catch (e) { /* 忽略 */ }
    }
    const img = $('m-image');
    const cv = $('m-canvas');
    if (img) {
      img.style.display = 'block';
    try { ViewerHub.set(null); } catch (e) { /* 忽略 */ }
      img.src = f.url;
      img.onerror = () => {
        $('m-overlay').classList.remove('hide');
        $('m-overlay').style.whiteSpace = 'pre-line';
        $('m-overlay').textContent = '图片加载失败：' + f.url;
      };
      img.onload = () => {
        $('m-overlay').classList.add('hide');
        $('m-status').textContent = f.name + ' · 服务端渲染 '
          + img.naturalWidth + '×' + img.naturalHeight
          + (img.naturalWidth >= 900 ? '（左=渲染 右=真值对比）' : '');
      };
    }
    if (cv) cv.style.display = 'none';
    return;
  }
  // 3D 类产物：确保 canvas 可见、图片隐藏
  const img0 = $('m-image');
  if (img0) { img0.style.display = 'none'; img0.removeAttribute('src'); }
  const cv0 = $('m-canvas');
  if (cv0) cv0.style.display = 'block';

  $('m-overlay').classList.remove('hide');
  $('m-overlay').textContent = '载入 ' + f.name + ' …';
  // 到这里说明是 glb / 普通 ply（高斯与图片在前面已分流）⇒ 现在才需要 WebGL1
  if (!ensureViewer()) {
    $('m-status').textContent = '查看器不可用';
    return;
  }
  try {
    // ⚠️ 不能只看文件名后缀判断格式：`gauss3d` 条目的显示名是
    // "point_cloud.ply（完整 59 维）"，`endsWith('.ply')` 为 false，
    // 会被误当成 glb 去解析（然后报含糊的错）。
    // 优先用后端给的 `kind`；它是权威来源。
    const low = String(f.name || '').toLowerCase();
    const kind = (f.kind === 'gauss3d' || low.endsWith('.splat') || f.kind === 'splat')
      ? 'splat'
      : ((low.endsWith('.ply') || f.url.toLowerCase().includes('.ply')) ? 'ply' : 'glb');
    const info = await window.GsViewer.load(f.url, kind);
    showModelInfo(f.name, info);
  } catch (e) {
    // 失败必须显示出来 —— 之前只把 overlay 留着，用户看到的就是一片黑，
    // 而真正的原因（哪个 URL 404、缺哪个脚本）只出现在浏览器控制台里。
    // 这里把 **URL 和格式**都写进界面，让"黑屏"自带诊断信息。
    $('m-overlay').classList.remove('hide');
    // 需要保留换行（多行诊断信息），否则会被 HTML 折叠成一行
    $('m-overlay').style.whiteSpace = 'pre-line';
    $('m-overlay').textContent = '载入失败：' + e.message
      + '\n格式：' + (f.kind || '?') + '\n地址：' + f.url
      + '\n\n自查：打开 /api/selftest?job=' + encodeURIComponent($('m-job').value || '')
      + ' 看服务端报告';
    $('m-status').textContent = '失败：' + e.message;
  }
}

/** 统一显示"载入成功"的信息（远端产物与本地文件共用） */
function showModelInfo(name, info) {
  $('m-overlay').classList.add('hide');
  const parts = [`${info.vertices} 顶点`];
  if (info.triangles) parts.push(`${info.triangles} 面`);
  if (info.gaussians) {
    // 说明渲染形态与信息维度 —— 用户据此知道"看到的是不是真实高斯"
    parts.push(`${info.gaussians} 高斯`);
    if (info.mode === 'gauss3d') {
      parts.push(`泼溅渲染（SH ${info.shDegree ?? 0} 阶${info.hasSH ? '，视角着色' : ''}）`);
    } else if (info.mode === 'splat') {
      parts.push('泼溅渲染（无 SH，`.splat` 仅 11 维）');
    }
  }
  // 降级必须**显式告知**：泼溅不可用时我们退回点云（而不是黑屏），
  // 但用户需要知道"现在看到的不是真实高斯"，否则会把降级当成模型质量差。
  if (info.degraded) {
    parts.push(`⚠ 已降级为点云（${info.degradeReason || '泼溅不可用'}）`);
  }
  if (info.textured) parts.push('带贴图');
  else if (info.colors) parts.push('顶点色');
  if (info.normals) parts.push('法线');
  // ---- 黑屏自检：画完抽查像素，亮点为 0 就明确报出来 ----
  //
  // 泼溅失败**不一定抛异常**：32 位索引扩展缺失、几何被投到屏幕外，
  // 都只会得到"一片黑 + 控制台干净"。用户报"还是黑屏"时，
  // 没有这条信息就只能靠猜。这里把结果直接写进状态栏。
  if (info.pixels && info.pixels.blank === true) {
    parts.push('⚠ 绘制后画面全黑（' + (info.pixels.sampled || 0)
      + ' 个采样点无一亮点）'
      + (info.indexed === false ? '；索引扩展缺失，已走无索引路径' : ''));
  } else if (info.pixels && info.pixels.blank === null && info.pixels.error) {
    parts.push('像素自检不可用：' + info.pixels.error);
  }
  $('m-status').textContent = name + ' · ' + parts.join(' · ');
}

/**
 * 直接打开本地 .ply / .glb 文件。
 *
 * 为什么需要：用户手上常有**不在作业目录里**的产物 —— 例如从别处拿到的
 * 点云 PLY、或 Blender 导出的实感网格 GLB。要求先"跑一个作业"才能看是不合理的。
 * 这里用 FileReader 读成 ArrayBuffer 后走 `GsViewer.loadBuffer()`，
 * 与远端产物**共用同一条解析路径**（避免两套解析逻辑各自出 bug）。
 * 文件不会上传：全程在浏览器内存里。
 */
function initLocalFilePicker() {
  const input = $('m-file');
  if (!input) return;
  input.onchange = async () => {
    const f = input.files && input.files[0];
    if (!f) return;
    const low = f.name.toLowerCase();
    const info = $('m-file-info');
    if (!low.endsWith('.ply') && !low.endsWith('.glb') && !low.endsWith('.splat')) {
      // 界面层就挡掉不支持的格式，不要让查看器抛含糊错误
      if (info) info.textContent = '不支持该格式（仅 .ply / .glb / .splat）：' + f.name;
      return;
    }
    if (info) {
      info.textContent = `读取中：${f.name}（${(f.size / 1048576).toFixed(2)} MB）…`;
    }
    $('m-overlay').classList.remove('hide');
    $('m-overlay').textContent = '读取 ' + f.name + ' …';
    try {
      const buf = await f.arrayBuffer();
      const kind = low.endsWith('.splat') ? 'splat' : (low.endsWith('.ply') ? 'ply' : 'glb');
      const r = await window.GsViewer.loadBuffer(buf, f.name, kind);
      showModelInfo(f.name, r);
      if (info) {
        info.textContent = `已载入本地文件 ${f.name}`
          + `（${(f.size / 1048576).toFixed(2)} MB，${r.vertices} 顶点`
          + (r.triangles ? `，${r.triangles} 面` : '')
          + (r.colors ? '，含顶点色' : '')
          + (r.normals ? '，含法线' : '') + '）';
      }
    } catch (e) {
      $('m-overlay').textContent = '打开失败：' + e.message;
      $('m-status').textContent = '失败';
      if (info) info.textContent = '打开失败：' + e.message;
    } finally {
      input.value = '';   // 允许再次选择同一个文件
    }
  };

  const clear = $('btn-m-clear');
  if (clear) {
    clear.onclick = () => {
      window.GsViewer.clear();
      $('m-overlay').classList.remove('hide');
      $('m-overlay').textContent = '选择一个模型…';
      $('m-status').textContent = '';
      const info = $('m-file-info');
      if (info) info.textContent = '';
    };
  }
}

async function selectModelJob() {
  const jid = $('m-job').value;
  if (!jid) { MODELS = null; renderModeButtons(); renderModeFiles(); return; }
  $('m-overlay').classList.remove('hide');
  $('m-overlay').textContent = '读取产物清单…';
  try {
    MODELS = await api('/api/models/' + encodeURIComponent(jid));
    if (MODELS.complete === false && MODELS.note) {
      $('m-hint').textContent = '提示：' + MODELS.note
        + '（导出未完整结束，下面列出的是已完成的产物）';
    }
    if (!MODELS.counts?.[ACTIVE_MODE]) {
      ACTIVE_MODE = MODE_DEF.find((d) => (MODELS.counts?.[d.key] || 0) > 0)?.key || 'points';
    }
    renderModeButtons();
    renderModeFiles();
    if (!(MODELS.counts?.points || MODELS.counts?.mesh || MODELS.counts?.textured)) {
      $('m-overlay').textContent = '该作业还没有导出产物';
    }
  } catch (e) {
    MODELS = null;
    $('m-overlay').textContent = '读取失败：' + e.message;
    renderModeButtons(); renderModeFiles();
  }
}

$('btn-m-reload').onclick = loadModelJobs;
$('m-job').onchange = selectModelJob;
// 三个按钮统一经 `ViewerHub` 路由（2026-10-25）：
//   * 显示 3DGS（参考渲染器）时，"重置视角"交给参考库的 resetCamera；
//     线框/旋转模式对它无意义 ⇒ 按钮**禁用**并在 title 里说明原因；
//   * 显示 glb/网格（自研 WebGL）时三个按钮照常。
// 这样不会再出现"命令打到另一个（隐藏的）渲染器"这种交互冲突。
$('btn-m-reset').onclick = () => {
  if (!ViewerHub.active && !ensureViewer()) return;
  ViewerHub.reset();
};
$('btn-m-wire').onclick = () => {
  if (!ViewerHub.supports('wire')) return;          // 按钮此时是禁用的，双保险
  if (!ensureViewer()) return;
  const on = ViewerHub.toggleWire();
  $('btn-m-wire').textContent = on ? '关闭线框' : '切换线框';
};
// 旋转模式切换：'free'（可越过顶/底继续翻，默认）| 'clamp'（到顶底停住）
// 两种都**不会中途反向**（那才是"卡住"）；差别只在极点处停住还是翻过去。
if ($('btn-m-rotmode')) {
  $('btn-m-rotmode').onclick = () => {
    if (!ViewerHub.supports('rotmode')) return;     // 同上，双保险
    if (!ensureViewer()) return;
    const cur = ViewerHub.rotMode() || 'clamp';
    const next = cur === 'free' ? 'clamp' : 'free';
    ViewerHub.setRotMode(next);
    $('btn-m-rotmode').textContent = next === 'free' ? '旋转模式：自由翻转' : '旋转模式：到顶底停住';
  };
}
// 首屏（还没加载任何模型）时三个按钮都不该可用
try { ViewerHub.syncButtons(); } catch (e) { /* 忽略 */ }

/* ------------------------------------------------------------------ 启动 */
(async function boot() {
  await health();
  await loadStages();
  await loadOpacityMap();   // 必须在 loadConfig 之前：表单要用到映射
  await loadCaptureMap();   // 同上：拍摄模式也要先拿到映射
  await refreshJobs();
  await loadConfig();
  await loadCreds();
  setInterval(refreshJobs, 5000);
})();
