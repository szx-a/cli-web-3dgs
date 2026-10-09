/* gsflow GUI —— 无框架、无 CDN 依赖，离线可用。 */
'use strict';

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
function collectParams() {
  const p = {};
  const put = (k, v) => { if (v !== '' && v !== null && v !== undefined) p[k] = v; };
  put('mask.backend', $('p-mask-backend').value);
  put('mask.mode', $('p-mask-mode').value);
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
      el.innerHTML = '产物：' + arts.slice(0, 12).map((a) => `<a href="/api/artifacts/${jobId}/${encodeURIComponent(a.replace(/\\\\/g, '/'))}" target="_blank">${a}</a>`).join(' · ');
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
  'p-frames-fps', 'p-frames-long', 'p-colmap-matcher', 'p-train-backend', 'p-train-iters',
  'p-train-res', 'p-train-cap', 'p-train-densify', 'p-train-maskloss',
  'p-export-formats', 'p-mesh-method', 'p-mesh-depth', 'p-mesh-tris', 'p-mesh-smooth',
  'p-mesh-width', 'p-mesh-views', 'p-mesh-densq', 'p-mesh-texture', 'p-mesh-texsize',
].forEach((id) => { const el = $(id); if (el) el.addEventListener('input', updateCliPreview); });

/* ------------------------------------------------------------------ 启动 */
(async function boot() {
  await health();
  await loadStages();
  await refreshJobs();
  await loadConfig();
  await loadCreds();
  setInterval(refreshJobs, 5000);
})();
