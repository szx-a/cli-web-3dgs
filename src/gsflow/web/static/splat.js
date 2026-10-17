/* ============================================================================
 * splat.js —— 高斯泼溅渲染器（让查看器显示**真正的高斯**而不是点云）
 *
 * ## 为什么需要它（用户反馈的直接原因）
 *
 * 用户说"GUI 查看器里的模型比 preview_front.png 差很多"。查清后发现不是渲染质量问题，
 * 而是**渲染的东西不对**：
 *   * `preview_*.png` 是 gsplat 泼溅渲染（位置 + 3×3 协方差 + 不透明度）；
 *   * 查看器此前只解析 glTF POINTS / PLY 点云，每个高斯被压成 `x,y,z,r,g,b`
 *     —— 丢了协方差、不透明度、SH ⇒ 必然是"稀疏点阵、有空隙、颜色扁平"。
 *
 * ⇒ 把 59 参数的高斯降维成 6 参数的点，当然差很多。
 *
 * ## 本文件做什么
 *
 * 解析 `.splat`（32 字节/高斯）并用 **WebGL 实例化四边形** 渲染：
 *   1. 每个高斯在 CPU 端算好 **3×3 世界协方差** M = R·diag(s²)·Rᵀ（6 个分量上传）；
 *   2. 顶点着色器把它经视图旋转、再经透视投影的 **Jacobian** 投成 **2×2 屏幕协方差**；
 *   3. 对 2×2 做**特征分解**得到椭圆长短轴，把四边形顶点按轴拉伸 ⇒ 朝向各异的椭球；
 *   4. 片元着色器算 `alpha = exp(-0.5·r²) · opacity`，**预乘 alpha 混合**
 *      （`ONE, ONE_MINUS_SRC_ALPHA`）⇒ 连续表面而不是硬点；
 *   5. 每帧按视空间深度**桶排序**（O(n)，2048 桶），保证 back-to-front 混合正确。
 *
 * ⚠️ 关键约定（写错会让画面全黑或椭球全错）：
 *   * 视图空间里相机看 **-Z**（与 viewer.js 的 `m4viewFromQuat` 一致）⇒ 深度 = **-z**；
 *   * `.splat` 的四元数是 **w,x,y,z** 顺序；
 *   * 协方差里的尺度是**已取指数**的世界单位，要用 `s²`。
 * ==========================================================================*/

window.GsSplatRenderer = (function () {
  const VERT = [
    'attribute vec2 aCorner;',
    'attribute vec3 aCenter;',
    'attribute vec3 aCovA;',   // xx, xy, xz
    'attribute vec3 aCovB;',   // yy, yz, zz
    'attribute vec4 aColor;',
    'uniform mat4 uView;',
    'uniform mat4 uProj;',
    'uniform vec2 uFocal;',    // 像素焦距 fx, fy
    'uniform vec2 uViewport;', // 画布像素宽高
    'varying vec4 vColor;',
    'varying vec2 vPos;',
    'void main() {',
    '  vec4 cam = uView * vec4(aCenter, 1.0);',
    '  float z = -cam.z;',                      // 相机看 -Z ⇒ 正深度
    '  if (z <= 0.001) {',                      // 在相机背后 ⇒ 丢到裁剪空间外
    '    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);',
    '    vColor = vec4(0.0); vPos = vec2(0.0); return;',
    '  }',
    '  mat3 M = mat3(aCovA.x, aCovA.y, aCovA.z,',
    '                aCovA.y, aCovB.x, aCovB.y,',
    '                aCovA.z, aCovB.y, aCovB.z);',
    '  mat3 R = mat3(uView);',                  // 视图旋转（无缩放）
    // ⚠️ **不能用 `transpose()` / `inverse()`**：那是 GLSL ES 3.00（WebGL2）才有的函数，
    // 在 WebGL1 的 ES 1.00 里会直接编译失败：
    //     ERROR: 'transpose' : no matching overloaded function found
    // 这里改用"标量点积"等价形式，全程只用 `dot()` 与 `mat3 * vec3`（ES 1.00 都支持）：
    //   2×2 屏幕协方差 = J · cov · Jᵀ，其中 cov = R·M·Rᵀ
    //   利用 (Rᵀw)·(M(Rᵀw')) = wᵀ·cov·w'，把 J 的两行投影到视图空间再乘 M。
    '  vec3 j0 = vec3(uFocal.x / z, 0.0, -uFocal.x * cam.x / (z * z));',
    '  vec3 j1 = vec3(0.0, uFocal.y / z, -uFocal.y * cam.y / (z * z));',
    // Rᵀ·j：`R[i]` 取的是**第 i 列**，所以第 i 个分量就是 dot(R[i], j)
    '  vec3 u0 = vec3(dot(R[0], j0), dot(R[1], j0), dot(R[2], j0));',
    '  vec3 u1 = vec3(dot(R[0], j1), dot(R[1], j1), dot(R[2], j1));',
    '  vec3 Mu0 = M * u0;',
    '  vec3 Mu1 = M * u1;',
    // 低通：给极小方差加 0.3 px²，避免亚像素高斯闪烁（Mip-Splatting 最简形式）
    '  float a = dot(u0, Mu0) + 0.3;',
    '  float b = dot(u0, Mu1);',
    '  float c = dot(u1, Mu1) + 0.3;',
    '  float mid = 0.5 * (a + c);',
    '  float disc = sqrt(max(0.0, mid * mid - (a * c - b * b)));',
    '  float l1 = mid + disc;',                 // 较大方差（px²）
    '  float l2 = max(mid - disc, 0.01);',
    '  vec2 v1 = (abs(b) > 1e-6) ? normalize(vec2(b, l1 - a))',
    '                            : ((a >= c) ? vec2(1.0, 0.0) : vec2(0.0, 1.0));',
    '  vec2 v2 = vec2(-v1.y, v1.x);',
    '  vec2 axis1 = v1 * sqrt(l1);',
    '  vec2 axis2 = v2 * sqrt(l2);',
    '  vec2 offPx = aCorner.x * axis1 + aCorner.y * axis2;',  // 像素偏移
    '  vec4 clip = uProj * cam;',
    '  clip.xy += (offPx / (uViewport * 0.5)) * clip.w;',     // 转回裁剪空间
    '  gl_Position = clip;',
    '  vPos = aCorner;',
    '  vColor = aColor;',
    '}',
  ].join('\n');

  const FRAG = [
    'precision mediump float;',
    'varying vec4 vColor;',
    'varying vec2 vPos;',
    'void main() {',
    '  float r2 = dot(vPos, vPos);',
    '  float A = exp(-0.5 * r2) * vColor.a;',
    '  if (A < 0.0039) discard;',               // < 1/255 直接丢
    '  gl_FragColor = vec4(vColor.rgb * A, A);',// 预乘 alpha
    '}',
  ].join('\n');

  function compile(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error('着色器编译失败：' + gl.getShaderInfoLog(s));
    }
    return s;
  }

  /** 由四元数 + 尺度算 3×3 世界协方差，返回 [xx, xy, xz, yy, yz, zz]。 */
  function covFromScaleQuat(sx, sy, sz, qw, qx, qy, qz) {
    // 旋转矩阵（列主序 R[col][row] 用 r00..r22 平铺表达）
    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;
    const r00 = 1 - 2 * (yy + zz), r01 = 2 * (xy - wz), r02 = 2 * (xz + wy);
    const r10 = 2 * (xy + wz), r11 = 1 - 2 * (xx + zz), r12 = 2 * (yz - wx);
    const r20 = 2 * (xz - wy), r21 = 2 * (yz + wx), r22 = 1 - 2 * (xx + yy);
    const s0 = sx * sx, s1 = sy * sy, s2 = sz * sz;
    // M = R · diag(s²) · Rᵀ
    const m00 = r00 * r00 * s0 + r01 * r01 * s1 + r02 * r02 * s2;
    const m01 = r00 * r10 * s0 + r01 * r11 * s1 + r02 * r12 * s2;
    const m02 = r00 * r20 * s0 + r01 * r21 * s1 + r02 * r22 * s2;
    const m11 = r10 * r10 * s0 + r11 * r11 * s1 + r12 * r12 * s2;
    const m12 = r10 * r20 * s0 + r11 * r21 * s1 + r12 * r22 * s2;
    const m22 = r20 * r20 * s0 + r21 * r21 * s1 + r22 * r22 * s2;
    return [m00, m01, m02, m11, m12, m22];
  }

  /**
   * 由**完整 3DGS 属性**构建（不丢信息维度）。
   *
   * 完整的 3DGS 每个高斯有 **59 个参数**：
   *   位置 3 + 尺度 3 + 旋转 4 + 不透明度 1 + SH **48**（3 DC + 45 高阶，deg=3）
   *
   * `.splat` 格式只存 11 个（位置/尺度/旋转/不透明度），**丢掉 SH 的 48 维**
   * ⇒ 丢失视角相关颜色（反光、菲涅尔、方向性明暗）。
   * 本函数直接接收 59 维全集，SH 在 `draw()` 里按**当前视线方向**求值，
   * 与训练/预览图（gsplat 渲染）看到的是同一个模型。
   *
   * 参数（都是扁平 TypedArray）：
   *   pos(n*3) 位置；scale(n*3) **log 尺度**（内部取 exp）；
   *   quat(n*4) 四元数 **w,x,y,z**（内部归一化）；opacity(n) **logit**（内部 sigmoid）；
   *   shDC(n*3)；shRest(n*45) 或 null；shDegree(0..3)。
   */
  function buildFromGauss3D(g) {
    const n = Math.floor(g.pos.length / 3);
    const center = new Float32Array(n * 3);
    const cov = new Float32Array(n * 6);
    const dirColor = new Float32Array(n * 4);   // 仅 DC 时的颜色（RGBA）
    const sh = g.shRest ? new Float32Array(n * 48) : null;  // 16 基 × 3 通道

    const SH_C0 = 0.28209479177387814;
    for (let i = 0; i < n; i++) {
      center[i * 3] = g.pos[i * 3];
      center[i * 3 + 1] = g.pos[i * 3 + 1];
      center[i * 3 + 2] = g.pos[i * 3 + 2];

      const sx = Math.exp(g.scale[i * 3]);
      const sy = Math.exp(g.scale[i * 3 + 1]);
      const sz = Math.exp(g.scale[i * 3 + 2]);
      let qw = g.quat[i * 4], qx = g.quat[i * 4 + 1];
      let qy = g.quat[i * 4 + 2], qz = g.quat[i * 4 + 3];
      const ql = Math.hypot(qw, qx, qy, qz) || 1;
      qw /= ql; qx /= ql; qy /= ql; qz /= ql;
      const c = covFromScaleQuat(sx, sy, sz, qw, qx, qy, qz);
      for (let k = 0; k < 6; k++) cov[i * 6 + k] = c[k];

      const op = 1 / (1 + Math.exp(-g.opacity[i]));
      dirColor[i * 4] = Math.min(1, Math.max(0, 0.5 + SH_C0 * g.shDC[i * 3]));
      dirColor[i * 4 + 1] = Math.min(1, Math.max(0, 0.5 + SH_C0 * g.shDC[i * 3 + 1]));
      dirColor[i * 4 + 2] = Math.min(1, Math.max(0, 0.5 + SH_C0 * g.shDC[i * 3 + 2]));
      dirColor[i * 4 + 3] = Math.min(1, Math.max(0, op));

      if (sh) {
        // SH 布局：基 0 = f_dc，基 1..15 来自 f_rest。
        // ⚠️ `f_rest` 是**通道主序**：先 R 的 15 个基，再 G，再 B。
        sh[i * 48] = g.shDC[i * 3];
        sh[i * 48 + 16] = g.shDC[i * 3 + 1];
        sh[i * 48 + 32] = g.shDC[i * 3 + 2];
        for (let ch = 0; ch < 3; ch++) {
          for (let k = 1; k < 16; k++) {
            sh[i * 48 + ch * 16 + k] = g.shRest[i * 45 + ch * 15 + (k - 1)];
          }
        }
      }
    }
    const bounds = robustBounds(center, n);
    return { count: n, center, cov, color: dirColor, sh,
             shDegree: g.shDegree | 0, bounds, hasSH: !!sh };
  }

  /** 稳健取景：中位数中心 + 95 分位距离（避免离群点把相机拉太远）。 */
  function robustBounds(center, n) {
    const cx = new Float64Array(n), cy = new Float64Array(n), cz = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      cx[i] = center[i * 3]; cy[i] = center[i * 3 + 1]; cz[i] = center[i * 3 + 2];
    }
    const med = (arr) => {
      const a = Float64Array.from(arr).sort();
      const k = a.length >> 1;
      return a.length % 2 ? a[k] : 0.5 * (a[k - 1] + a[k]);
    };
    const mc = [med(cx), med(cy), med(cz)];
    const dist = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      dist[i] = Math.hypot(cx[i] - mc[0], cy[i] - mc[1], cz[i] - mc[2]);
    }
    const sorted = Float64Array.from(dist).sort();
    const q95 = sorted[Math.min(n - 1, Math.max(0, Math.floor(0.95 * (n - 1))))];
    let radius = q95;
    if (!(radius > 1e-6)) {
      let mnx = Infinity, mny = Infinity, mnz = Infinity;
      let mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
      for (let i = 0; i < n; i++) {
        if (cx[i] < mnx) mnx = cx[i]; if (cx[i] > mxx) mxx = cx[i];
        if (cy[i] < mny) mny = cy[i]; if (cy[i] > mxy) mxy = cy[i];
        if (cz[i] < mnz) mnz = cz[i]; if (cz[i] > mxz) mxz = cz[i];
      }
      radius = Math.max(1e-6, 0.5 * Math.hypot(mxx - mnx, mxy - mny, mxz - mnz));
    }
    return { center: mc, radius, robust: true, q95 };
  }

  /** 3DGS 的 SH 基（与官方 `sh_utils.py::eval_sh` 的常数逐项一致）。 */
  const SH_C1 = 0.4886025119029199;
  const SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                 -1.0925484305920792, 0.5462742152960396];
  const SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                 0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                 -0.5900435899266435];

  /**
   * 求值 SH（deg≤3），写入 `color`（RGBA 数组，原地）。
   * `eye` 是相机位置；方向取 **(高斯中心 - 相机)** 归一化 —— 与官方 3DGS 一致。
   */
  function evalSHInto(sh, deg, center, n, eye, color) {
    const SH_C0 = 0.28209479177387814;
    for (let i = 0; i < n; i++) {
      let dx = center[i * 3] - eye[0];
      let dy = center[i * 3 + 1] - eye[1];
      let dz = center[i * 3 + 2] - eye[2];
      const dl = Math.hypot(dx, dy, dz) || 1;
      dx /= dl; dy /= dl; dz /= dl;
      const xx = dx * dx, yy = dy * dy, zz = dz * dz;
      const xy = dx * dy, yz = dy * dz, xz = dx * dz;
      const o = i * 48;
      for (let ch = 0; ch < 3; ch++) {
        const b = o + ch * 16;
        let v = SH_C0 * sh[b];
        if (deg > 0) {
          v += -SH_C1 * dy * sh[b + 1] + SH_C1 * dz * sh[b + 2] - SH_C1 * dx * sh[b + 3];
          if (deg > 1) {
            v += SH_C2[0] * xy * sh[b + 4] + SH_C2[1] * yz * sh[b + 5]
               + SH_C2[2] * (2 * zz - xx - yy) * sh[b + 6]
               + SH_C2[3] * xz * sh[b + 7] + SH_C2[4] * (xx - yy) * sh[b + 8];
            if (deg > 2) {
              v += SH_C3[0] * dy * (3 * xx - yy) * sh[b + 9]
                 + SH_C3[1] * xy * dz * sh[b + 10]
                 + SH_C3[2] * dy * (4 * zz - xx - yy) * sh[b + 11]
                 + SH_C3[3] * dz * (2 * zz - 3 * xx - 3 * yy) * sh[b + 12]
                 + SH_C3[4] * dx * (4 * zz - xx - yy) * sh[b + 13]
                 + SH_C3[5] * dz * (xx - yy) * sh[b + 14]
                 + SH_C3[6] * dx * (xx - 3 * yy) * sh[b + 15];
            }
          }
        }
        color[i * 4 + ch] = Math.min(1, Math.max(0, 0.5 + v));
      }
    }
  }

  /** 解析 `.splat`：32 字节/条 = pos3(f32) + scale3(f32) + rgba4(u8) + quat4(u8)。 */
  function parseSplat(buf) {
    const n = Math.floor(buf.byteLength / 32);
    if (n <= 0) throw new Error('splat 文件为空');
    const u8 = new Uint8Array(buf);
    const f32 = new Float32Array(buf);
    const center = new Float32Array(n * 3);
    const cov = new Float32Array(n * 6);
    const color = new Float32Array(n * 4);
    let maxScale = 0;
    for (let i = 0; i < n; i++) {
      const fb = i * 8;             // 8 个 float32 槽位
      const bb = i * 32;            // 字节偏移
      center[i * 3] = f32[fb];
      center[i * 3 + 1] = f32[fb + 1];
      center[i * 3 + 2] = f32[fb + 2];
      const sx = f32[fb + 3], sy = f32[fb + 4], sz = f32[fb + 5];
      if (sx > maxScale) maxScale = sx;
      if (sy > maxScale) maxScale = sy;
      if (sz > maxScale) maxScale = sz;
      // 四元数字节解码：q = (b - 128) / 128，顺序 w,x,y,z
      let qw = (u8[bb + 28] - 128) / 128;
      let qx = (u8[bb + 29] - 128) / 128;
      let qy = (u8[bb + 30] - 128) / 128;
      let qz = (u8[bb + 31] - 128) / 128;
      const ql = Math.hypot(qw, qx, qy, qz) || 1;
      qw /= ql; qx /= ql; qy /= ql; qz /= ql;
      const c = covFromScaleQuat(sx, sy, sz, qw, qx, qy, qz);
      for (let k = 0; k < 6; k++) cov[i * 6 + k] = c[k];
      color[i * 4] = u8[bb + 24] / 255;
      color[i * 4 + 1] = u8[bb + 25] / 255;
      color[i * 4 + 2] = u8[bb + 26] / 255;
      color[i * 4 + 3] = u8[bb + 27] / 255;
    }
    // 取景包围球：用**稳健半径**（中位数中心 + 95 分位距离），不用包围盒。
    // 实测 img3-crop 的包围盒半径 4.76，而 95% 的高斯都在距中心 0.84 内
    // —— 只有 0.61% 的离群点就把半径撑大 5.65 倍，相机会被拉太远。
    const bounds = robustBounds(center, n);
    return { count: n, center, cov, color, maxScale, bounds, hasSH: false };
  }

  /** 从一个**数据对象**（parseSplat 或 buildFromGauss3D 的结果）建立 GL 资源。 */
  function createFrom(gl, data) {
    const ext = gl.getExtension('ANGLE_instanced_arrays');
    if (!ext) throw new Error('浏览器不支持 ANGLE_instanced_arrays，无法渲染泼溅');
    const n = data.count;

    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('着色器链接失败：' + gl.getProgramInfoLog(prog));
    }
    const aCorner = gl.getAttribLocation(prog, 'aCorner');
    const aCenter = gl.getAttribLocation(prog, 'aCenter');
    const aCovA = gl.getAttribLocation(prog, 'aCovA');
    const aCovB = gl.getAttribLocation(prog, 'aCovB');
    const aColor = gl.getAttribLocation(prog, 'aColor');
    const U = {
      uView: gl.getUniformLocation(prog, 'uView'),
      uProj: gl.getUniformLocation(prog, 'uProj'),
      uFocal: gl.getUniformLocation(prog, 'uFocal'),
      uViewport: gl.getUniformLocation(prog, 'uViewport'),
    };

    // 四边形角点：TRIANGLE_STRIP 的 4 个顶点，±2σ 覆盖
    const quad = new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]);
    const qbuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, qbuf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    // 顺序索引（每帧按深度重排后上传）
    const idx = new Uint32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    const ibuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.DYNAMIC_DRAW);

    // 把 center/cov/color 预先按"索引顺序"重排的缓冲（重排用 CPU 侧数组）
    const cbuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cbuf);
    gl.bufferData(gl.ARRAY_BUFFER, data.center, gl.STATIC_DRAW);
    const covAbuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, covAbuf);
    gl.bufferData(gl.ARRAY_BUFFER, data.cov, gl.STATIC_DRAW);
    const colbuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colbuf);
    gl.bufferData(gl.ARRAY_BUFFER, data.color, gl.DYNAMIC_DRAW);

    // 桶排序辅助
    const BUCKETS = 2048;
    const depth = new Float32Array(n);
    const counts = new Uint32Array(BUCKETS + 1);
    let lastEye = null, lastQ = null;

    function sortByDepth(view, eye) {
      // 深度 = 视空间 -z（相机看 -Z）。桶排序 O(n)，避免 117k 元素比较排序卡顿。
      let dmin = Infinity, dmax = -Infinity;
      const m = view;
      for (let i = 0; i < n; i++) {
        const x = data.center[i * 3], y = data.center[i * 3 + 1], z = data.center[i * 3 + 2];
        const vz = m[2] * x + m[6] * y + m[10] * z + m[14];
        const d = -vz;
        depth[i] = d;
        if (d < dmin) dmin = d;
        if (d > dmax) dmax = d;
      }
      if (!isFinite(dmin) || dmax <= dmin) return;
      counts.fill(0);
      const scale = (BUCKETS - 1) / (dmax - dmin);
      for (let i = 0; i < n; i++) {
        counts[Math.min(BUCKETS - 1, (depth[i] - dmin) * scale | 0)]++;
      }
      // 前缀和（从远到近画：远 = 大深度 ⇒ 桶 0 放最远）
      let acc = 0;
      for (let b = BUCKETS - 1; b >= 0; b--) { const c = counts[b]; counts[b] = acc; acc += c; }
      for (let i = 0; i < n; i++) {
        const b = Math.min(BUCKETS - 1, (depth[i] - dmin) * scale | 0);
        idx[counts[b]++] = i;
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf);
      gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, idx);
    }

    function bindAttrib(buffer, loc, size, stride, offset) {
      if (loc < 0) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride * 4, offset * 4);
      ext.vertexAttribDivisorANGLE(loc, 1);
    }

    function draw(view, proj, width, height, eye, qcam) {
      // 相机/朝向变了才重排（拖拽时每帧都会变，桶排序 O(n) 可承受）
      const changed = !lastEye || !lastQ
        || Math.abs(eye[0] - lastEye[0]) + Math.abs(eye[1] - lastEye[1]) + Math.abs(eye[2] - lastEye[2]) > 1e-6
        || Math.abs(qcam[0] - lastQ[0]) + Math.abs(qcam[1] - lastQ[1]) + Math.abs(qcam[2] - lastQ[2]) + Math.abs(qcam[3] - lastQ[3]) > 1e-6;
      if (changed) {
        sortByDepth(view, eye);
        // ---- SH 视角相关着色（**不丢信息维度**的关键）----
        // 完整 3DGS 的颜色是 SH(deg 3) 的 16 个基 × 3 通道，依赖**视线方向**。
        // `.splat` 只有 DC ⇒ 只能画平均色。这里对从 PLY 读入的完整 SH 求值，
        // 于是查看器看到的是与训练/预览图（gsplat 渲染）**同一个模型**的外观。
        // 只在相机变化时算（与排序同频），117k × 48 次乘加 ≈ 十几毫秒，可接受。
        if (data.hasSH && data.sh) {
          evalSHInto(data.sh, data.shDegree, data.center, n, eye, data.color);
          gl.bindBuffer(gl.ARRAY_BUFFER, colbuf);
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, data.color);
        }
        lastEye = eye.slice();
        lastQ = qcam.slice();
      }

      gl.useProgram(prog);
      gl.uniformMatrix4fv(U.uView, false, view);
      gl.uniformMatrix4fv(U.uProj, false, proj);
      // 像素焦距：proj[0]=1/(aspect*tan(fov/2)), proj[5]=1/tan(fov/2)
      gl.uniform2f(U.uFocal, proj[0] * width * 0.5, proj[5] * height * 0.5);
      gl.uniform2f(U.uViewport, width, height);

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);

      gl.bindBuffer(gl.ARRAY_BUFFER, qbuf);
      // ⚠️ location 可能为 -1（attribute 在 GLSL 里未被使用 / 被优化掉）。
      // 对 -1 调 enableVertexAttribArray 会产生 INVALID_VALUE 且 dirty 整个 GL 状态，
      // 后续真正的错误就被掩盖了。这里必须判。
      if (aCorner >= 0) {
        gl.enableVertexAttribArray(aCorner);
        gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);
        ext.vertexAttribDivisorANGLE(aCorner, 0);
      }

      bindAttrib(cbuf, aCenter, 3, 3, 0);
      bindAttrib(covAbuf, aCovA, 3, 6, 0);
      bindAttrib(covAbuf, aCovB, 3, 6, 3);
      bindAttrib(colbuf, aColor, 4, 4, 0);

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf);
      ext.drawElementsInstancedANGLE(gl.TRIANGLE_STRIP, 4, gl.UNSIGNED_INT, 0, n);

      gl.depthMask(true);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
    }

    return {
      count: n,
      maxScale: data.maxScale,
      bounds: data.bounds,
      draw,
      program: prog,
    };
  }

  /** 从 `.splat`（32B/高斯、无 SH）建立。 */
  function create(gl, buf) {
    return createFrom(gl, parseSplat(buf));
  }

  /** 从**完整 3DGS 属性**（59 维/高斯、含 SH）建立。 */
  function createFromGauss3D(gl, g) {
    return createFrom(gl, buildFromGauss3D(g));
  }

  return { create, createFrom, createFromGauss3D, parseSplat,
           buildFromGauss3D, evalSHInto, covFromScaleQuat, robustBounds };
})();
