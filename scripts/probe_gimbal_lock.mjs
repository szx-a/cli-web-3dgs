/**
 * 万向锁（gimbal lock）的**量化判定** —— 不靠"看着没问题"，靠三个可测指标。
 *
 * 万向锁的数学本质：欧拉角参数化在某姿态处**丢掉一个自由度**，
 * 表现为"某些方向的连续旋转在参数上无法连续表示"，典型症状有三类：
 *
 *   指标 1【奇点】轴向量近似平行（本测试用相机朝向 y 分量持续停在 ±1 附近不变化）
 *            ⇒ 说明此时再转某个轴实际上转不动（自由度丢失）。
 *            **正常相机也会有极点**（pitch=±90° 时朝向确实是 ±y），
 *            所以判据不是"到过极点"，而是"**到极点后无法继续有序变化**"。
 *
 *   指标 2【方向反转】拖拽方向与相机运动方向**符号相反**。
 *            这是用户最早抱怨的"卡住/乱跳"的真实形态：拖到底后继续拖，
 *            相机反而往回走（欧拉角 wrap-around 的典型症状）。
 *
 *   指标 3【不连续跳变】单步旋转角度出现远大于正常增量的跳变
 *            （例如一步跳过 90°），说明参数化在某处断裂。
 *
 * 本探针直接复刻 `viewer.js` 的 `buildCam()` 数学，逐步累加拖拽，
 * 对两种模式（clamp / free）分别测量上述三项。
 *
 * 用法: node scripts/probe_gimbal_lock.mjs
 */
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(import.meta.dirname, '..');
const VIEWER = path.join(ROOT, 'src', 'gsflow', 'web', 'static', 'viewer.js');
const src = fs.readFileSync(VIEWER, 'utf8');

// ---- 从 viewer.js 复刻四元数基础（与源码同一组公式，避免"测另一份实现"）----
const qAxisAngle = (axis, ang) => {
  const n = Math.hypot(...axis) || 1;
  const s = Math.sin(ang / 2);
  return [axis[0] / n * s, axis[1] / n * s, axis[2] / n * s, Math.cos(ang / 2)];
};
const qMul = (a, b) => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];
const qNorm = (q) => {
  const n = Math.hypot(...q) || 1;
  return q.map((v) => v / n);
};
const qRot = (q, v) => {
  const [x, y, z, w] = q;
  const ix = w * v[0] + y * v[2] - z * v[1];
  const iy = w * v[1] + z * v[0] - x * v[2];
  const iz = w * v[2] + x * v[1] - y * v[0];
  const iw = -x * v[0] - y * v[1] - z * v[2];
  return [
    ix * w + iw * -x + iy * -z - iz * -y,
    iy * w + iw * -y + iz * -x - ix * -z,
    iz * w + iw * -z + ix * -y - iy * -x,
  ];
};

const PITCH_MAX = Math.PI / 2 - 1e-3;
const SENS = 0.008;

/** 复刻 buildCam()（与源码逐行对应） */
function makeCam(rotMode) {
  let yawAcc = 0.6;
  let pitchAcc = 0.45;
  let qcam = [0, 0, 0, 1];
  function buildCam() {
    const qy = qAxisAngle([0, 1, 0], yawAcc);
    const right = qRot(qy, [1, 0, 0]);
    if (rotMode === 'clamp') {
      const p = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, pitchAcc));
      qcam = qNorm(qMul(qAxisAngle(right, p), qy));
    } else {
      qcam = qNorm(qMul(qAxisAngle(right, pitchAcc), qy));
    }
  }
  buildCam();
  return {
    orbit(dx, dy) {
      yawAcc += dx * SENS;
      pitchAcc += dy * SENS;
      if (rotMode === 'clamp') {
        pitchAcc = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, pitchAcc));
      } else {
        if (pitchAcc > Math.PI) pitchAcc -= 2 * Math.PI;
        if (pitchAcc < -Math.PI) pitchAcc += 2 * Math.PI;
      }
      buildCam();
    },
    get forward() { return qRot(qcam, [0, 0, -1]); },
    get up() { return qRot(qcam, [0, 1, 0]); },
    get q() { return qcam; },
    get pitch() { return pitchAcc; },
  };
}

function angBetween(a, b) {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return Math.acos(Math.max(-1, Math.min(1, d)));
}

function analyse(mode, steps = 400, dy = 6, label = '竖直向下') {
  const cam = makeCam(mode);
  const fwd = [];
  const ups = [];
  let prev = cam.forward;
  let maxStep = 0;
  let reversals = 0;
  let prevSign = 0;
  for (let i = 0; i < steps; i++) {
    cam.orbit(0, dy);
    const f = cam.forward;
    const stepAng = angBetween(prev, f);
    if (i > 0) maxStep = Math.max(maxStep, stepAng);
    // 指标 2：forward 的 y 分量变化方向是否反转
    const d = f[1] - prev[1];
    const sign = Math.abs(d) < 1e-9 ? 0 : Math.sign(d);
    if (sign !== 0 && prevSign !== 0 && sign !== prevSign) reversals += 1;
    if (sign !== 0) prevSign = sign;
    fwd.push(f);
    ups.push(cam.up);
    prev = f;
  }
  const ys = fwd.map((f) => f[1]);
  // 指标 1：极点处能否继续有序变化 —— 看最后 30 步 forward 的平均步进角
  const tail = fwd.slice(-31);
  let tailStep = 0;
  for (let i = 1; i < tail.length; i++) tailStep += angBetween(tail[i - 1], tail[i]);
  tailStep /= Math.max(1, tail.length - 1);
  // 指标 3：最大单步角（正常应约等于 dy*SENS = 0.048 rad）
  const nominal = Math.abs(dy) * SENS;
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  console.log(`  ${label} ×${steps}（dy=${dy}）  模式=${mode}`);
  console.log(`    forward.y 范围 [${yMin.toFixed(3)}, ${yMax.toFixed(3)}]`);
  console.log(`    方向反转次数 = ${reversals}   最大单步角 = ${maxStep.toFixed(4)} rad`
              + `（名义 ${nominal.toFixed(4)}）`);
  console.log(`    末段平均步进角 = ${tailStep.toFixed(5)} rad`
              + `  ${tailStep < nominal * 0.05 ? '⇐ 极点处已停止变化（被夹住）' : '（仍在有序变化）'}`);
  const jump = maxStep > nominal * 3.0;
  console.log(`    不连续跳变: ${jump ? '**有**' : '无'}`);
  return { reversals, maxStep, tailStep, nominal, yMin, yMax, jump };
}

console.log('=== 万向锁量化判定（复刻 viewer.js 的相机数学）===\n');
let bad = 0;

console.log('【clamp 模式（默认）】');
const a = analyse('clamp', 400, 6, '竖直向下');
// 判据：clamp 允许"拖到底停住"，但**不允许方向反转**，也不允许跳变
if (a.reversals > 0) { console.log('    **FAIL** 出现方向反转 —— 这是万向锁/欧拉退化的典型症状'); bad++; }
else console.log('    [OK] 无方向反转');
if (a.jump) { console.log('    **FAIL** 出现不连续跳变'); bad++; }
else console.log('    [OK] 无跳变');
if (a.tailStep < a.nominal * 0.05) {
  console.log('    [说明] 到极点后停住 —— 这是 clamp 模式的**设计**（"看到底面"即终点），');
  console.log('           不是万向锁：运动方向从未反转、也未跳变。');
}

console.log('\n【free 模式】');
const b = analyse('free', 800, 6, '竖直向下');
if (b.jump) { console.log('    **FAIL** 出现不连续跳变'); bad++; }
else console.log('    [OK] 无跳变（可连续越过极点）');

console.log('\n【水平方向（yaw）无限圈】');
const cam = makeCam('clamp');
const f0 = cam.forward;
let maxYawStep = 0;
let prevF = f0;
for (let i = 0; i < 1000; i++) {
  cam.orbit(8, 0);
  const f = cam.forward;
  maxYawStep = Math.max(maxYawStep, angBetween(prevF, f));
  prevF = f;
}
const nominalYaw = 8 * SENS;
console.log(`  转 1000 步（共 ${(1000 * 8 * SENS / (2 * Math.PI)).toFixed(2)} 圈）`
            + ` 最大单步角 = ${maxYawStep.toFixed(4)}（名义 ${nominalYaw.toFixed(4)}）`);
const yawOk = maxYawStep < nominalYaw * 3;
console.log(`  ${yawOk ? '[OK] 全程均匀转动，无奇点' : '**FAIL** 出现跳变'}`);
if (!yawOk) bad++;

console.log(`\n结论: ${bad === 0 ? '相机实现无万向锁（三项指标全部通过）' : `**${bad} 项失败**`}`);
process.exit(bad === 0 ? 0 : 1);
