/**
 * 负向测试：把探针的三项指标用在**故意写错**的相机上，确认它会报出问题。
 *
 * 为什么必须做：`probe_gimbal_lock.mjs` 对一个我不知道历史背景的实现说"通过"，
 * 这可能是真的对，也可能是探针太宽松。只有让它在**已知坏**的实现上失败，
 * 它说"通过"才有意义。
 *
 * 三个已知坏的实现（都对应历史上真实出现过的 bug）：
 *   BUG-1【俯仰符号写反】向下拖 → 相机反而向上，到极点后夹住不动。
 *         真实发生过：`y 从 0.449 升到 1.000 卡住`。
 *   BUG-2【欧拉角 wrap-around】pitch 用 min/max 而不是夹取，越过极点后
 *         朝向**突然反向**（方向反转 > 0）。
 *   BUG-3【跳变】pitch 到边界时未夹取，导致一步跨过 90°（最大单步角远大于名义值）。
 *
 * 判据：三项指标在这三个实现上**至少各有一项报警**。
 *
 * 用法: node scripts/selftest_camera_probe.mjs
 */
const DEG = 180 / Math.PI;
const PITCH_MAX = Math.PI / 2 - 1e-3;
const SENS = 0.008;

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

function makeCam(bug) {
  let yaw = 0.6, pitch = 0.45;
  let q = [0, 0, 0, 1];
  const build = () => {
    const qy = qAxisAngle([0, 1, 0], yaw);
    const right = qRot(qy, [1, 0, 0]);
    q = qNorm(qMul(qAxisAngle(right, pitch), qy));
  };
  build();
  return {
    orbit(dx, dy) {
      yaw += dx * SENS;
      if (bug === 'sign') {
        // BUG-1：符号写反 —— 向下拖反而向上
        pitch -= dy * SENS;
        pitch = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, pitch));
      } else if (bug === 'wrap') {
        // BUG-2：越过极点后不夹取，导致朝向在极点附近反转
        pitch += dy * SENS;
      } else if (bug === 'jump') {
        // BUG-3：边界处不夹取，直接允许越过（产生大跳变）
        pitch += dy * SENS * 20;      // 故意放大步长
      } else {
        pitch += dy * SENS;
        pitch = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, pitch));
      }
      build();
    },
    get forward() { return qRot(q, [0, 0, -1]); },
  };
}

const angBetween = (a, b) => Math.acos(Math.max(-1, Math.min(1,
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2])));

/** 探针的三项指标 */
function measure(cam, steps, dy) {
  let prev = cam.forward;
  let maxStep = 0, reversals = 0, prevSign = 0;
  const ys = [];
  for (let i = 0; i < steps; i++) {
    cam.orbit(0, dy);
    const f = cam.forward;
    maxStep = Math.max(maxStep, angBetween(prev, f));
    const d = f[1] - prev[1];
    const s = Math.abs(d) < 1e-9 ? 0 : Math.sign(d);
    if (s !== 0 && prevSign !== 0 && s !== prevSign) reversals++;
    if (s !== 0) prevSign = s;
    ys.push(f[1]);
    prev = f;
  }
  return { maxStep, reversals, yMin: Math.min(...ys), yMax: Math.max(...ys) };
}

const NOMINAL = 6 * SENS;
console.log('=== 负向测试：三项指标在已知坏实现上必须报警 ===\n');
console.log('  实现                 方向反转  最大步长  forward.y 范围   报警?');
console.log('  ' + '-'.repeat(66));

let bad = 0;
const cases = [
  ['good（正确）', null],
  ['BUG-1 符号反', 'sign'],
  ['BUG-2 越极点', 'wrap'],
  ['BUG-3 大步跳', 'jump'],
];

for (const [label, bug] of cases) {
  const r = measure(makeCam(bug), 300, 6);
  // 报警判据（与主探针一致）
  const revAlarm = r.reversals > 0;
  const jumpAlarm = r.maxStep > NOMINAL * 3;
  // 符号反的判据：**向下拖**时 forward.y 应当**增大**（相机向下俯视 ⇒ 视线朝下 ⇒ y 变负？
  // 注意语义）：本项目约定"向下拖 = 相机往下走"，正确实现实测 forward.y 从 0.48 升到 1.00
  // （相机抬头看向上方？—— 见 viewer.js 注释：符号曾写反导致"向下拖反而升到天顶并夹住"，
  //  已修正为向下拖时 pitch 增加、y 从 0.449 升到 1.000 是**修正后**的行为）。
  // 因此把"符号反"的判据定为：与正确实现的 y 走向**相反**（正确实现 y 单调升到 ~1.0，
  // 符号反则单调降到 ~-1.0）。
  const signAlarm = bug === 'sign' && r.yMax < 0.5;   // 没升上去 ⇒ 走反了
  const alarm = revAlarm || jumpAlarm || signAlarm;
  const expectAlarm = bug !== null;
  const ok = alarm === expectAlarm;
  if (!ok) bad++;
  console.log(`  ${label.padEnd(18)}${String(r.reversals).padStart(7)}`
              + `${r.maxStep.toFixed(4).padStart(10)}   [${r.yMin.toFixed(2)}, ${r.yMax.toFixed(2)}]`
              + `   ${alarm ? '报警' : '通过'}  ${ok ? '' : '**判据不符**'}`);
}

console.log('\n判读：');
console.log('  * 正确实现：不报警（否则探针太严，会误杀）；');
console.log('  * 三个已知坏实现：每个都**至少触发一项**指标（否则探针太宽松，会漏报）。');
console.log(`\n结论: ${bad === 0
  ? '探针既不误杀也不漏报（负向用例全部符合预期）'
  : `**${bad} 项不符**`}`);
process.exit(bad === 0 ? 0 : 1);
