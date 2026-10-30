# gsflow · AI 图像清理 + 3DGS 三维重建全流程（Windows）

> **一套流程核心，两个入口（CLI / Web GUI），行为完全一致、可断点续跑。**
> 视频/照片 → AI 前景分割与背景清理 → COLMAP 位姿恢复 → 3DGS 训练（gsplat）→ 网格/贴图/查看器导出 → 质量体检。

[![tests](https://img.shields.io/badge/tests-218%20passed-brightgreen)](#测试与闸门)
[![python](https://img.shields.io/badge/python-3.13-blue)](pyproject.toml)
[![cuda](https://img.shields.io/badge/CUDA-12.8%20(sm__120)-green)](#硬件与约束)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## 一、这是什么

把"拍一段物体视频 → 得到一个能看的 3D 模型"这件事**工程化**：

| 阶段 | 做什么 | 关键实现 |
|---|---|---|
| ① frames | 抽帧 / 图片导入 | ffmpeg；fps / 按张数 / 清晰度剔除 / 自动裁黑边 |
| ② mask | AI 前景分割 + 背景清理 | 后端可插拔（rembg / FastSAM / 经典算法）；**确定性拍摄模式判定**；掩码质量闸门 |
| ③ colmap | 稀疏重建（位姿） | COLMAP 4.2.1（顺序/穷举匹配 + mask 接入），失败回退 pycolmap |
| ④ train | 3DGS 训练 | gsplat 1.5.3；致密化/重置/剪枝；掩码复合损失；可选 2DGS 面元正则、LPIPS、抗锯齿 |
| ⑤ export | 导出 | 点云/网格 GLB、`.splat`、查看器 HTML、纹理烘焙（xatlas + 多视角采样） |
| ⑥ metrics | 质量报告 | 注册率、重投影误差、掩码内 PSNR/SSIM/LPIPS、白边、彩度 |

**设计红线**：流程核心在 `src/gsflow/pipeline` + `stages`，**不依赖 UI**；阶段之间只通过磁盘产物 +
`job.json` 通信，因此天然支持跳过 / 续跑 / 单阶段重跑。

---

## 二、实测数据（同一套代码，三份素材）

| 素材 | 帧数 | COLMAP | 稀疏点 | 重投影 | 掩码内 PSNR | 掩码内 SSIM | 全图 PSNR | 白边 | 彩度 |
|---|---|---|---|---|---|---|---|---|---|
| img3（406 张照片）| 406 | 406/406 | 8,753 | 1.08 px | 20.96 | 0.9477 | 3.42（黑底模式）| 0.051 | — |
| **img4（视频，平视）** | 100 | **98/98** | **18,638** | 0.478 px | **27.22** | 0.9571 | 18.68 | 0.282 | 0.063 |
| **img5（视频，俯视）** | 195 | **193/193** | **40,105** | 0.450 px | **26.48** | **0.9654** | 18.84 | 0.350 | 0.058 |

> **看掩码内指标，不要看全图 PSNR**：黑底模式下背景被合成成黑色，全图 PSNR 会把"蒙版是否正确"混进来。
> 白边（rim_bright）>0.10 说明有可见光晕；彩度（chroma）>0.12 说明有彩虹斑块。

**网格（img5-v1，TSDF 路线）**：278,810 面 → 水密化后 **282,088 面、边界边 3,278 → 0、封闭 ✔**。

**感知指标（同一把论文对齐的尺子）**：baseline **0.2502** → 每步 LPIPS 训练 **0.2481**（更低更好）。

---

## 三、创新与关键取舍（都是实测结论，不是论文口号）

1. **确定性拍摄模式判定**：同一素材两次跑出不同掩码模式（orbit vs turntable）曾导致 SSIM 0.9267 vs 0.9485。
   现在**先**用 mask 感知的检测（采样 12 帧比较背景差异）再合成，模式判定可复现。
2. **双掩码损失**：mask_loss_mode=composite + bg_alpha_reg，把"物体外不许有高斯"变成可优化项；
   同时保留原始掩码用于背景抑制（不侵蚀前景）。
3. **退火式"推实"正则**（opacity_crisp_reg=0.02 + warmup=0.7）：把"淡高斯云"推成实心表面，
   **`op>0.9` 15%→22%，SSIM 只 −0.0003**；全程施加会掉 0.0043 并出亮斑，所以必须退火。
4. **深度语义统一**：训练侧法线/失真正则从 `Σwz`（未归一化）改为 **ED（期望深度）**，与导出反投影一致。
   旧实现两者相差 1/alpha 倍 —— 各段都能跑，但**语义不自洽**（已加测试锁定）。
5. **网格水密化 + ED 精修**：边界边少 ≠ 封闭；fill_holes 按边界环补面（逆原面绕序），
   再用 ED 深度做**回溯式**顶点精修（ICP 风格阻尼，残差单调下降）。
   实测：**7 轮即收敛**，中位深度残差 −14.7%，表面 Chamfer 仅 **0.14% 场景尺度**，拓扑与观感不变；
   `mesh_refine_iters=300` 与 100 完全等价（`docs/03` G163）。
6. **负结果也写进文档**：LPIPS 每 16 步才算是**错的**（梯度与 L1/SSIM 打架，实测 LPIPS 反而变差），
   必须每步算；opacity_init=0.5 无效（淡高斯是目标函数的强吸引子）。
7. **一套"能定位问题"的质量指标族**：常规重建只报 PSNR/SSIM，而本项目的实测反复出现
   "指标很好、看起来却糊/薄/有白边"的情况。于是把观感拆成可计算的量：
   **轮廓光晕** $\text{rim}$、**轨道彩度** $\chi$、**不透明度分布**（实心度）、**掩码松紧** $\rho$、
   **网格封闭性**（边界边/非流形），并给出各自的判读阈值（见 3.7 与第七节）。

---

### 3.1 确定性拍摄模式判定（mask-aware）

转台（物体转、背景不动）与环绕（相机绕物体转、背景随视角变化）在数学上是**两类不同的相机模型假设**：
转台意味着"世界坐标系随手转"，直接把背景当静态会与相机位姿矛盾。用背景的**时间稳定性**判别：

$$s_b=\frac{\mathrm{median}_{u\notin M}\;\mathrm{Var}_t(I_t(u))}
{\mathrm{median}_{u\notin M}\;(I_t(u))^2},$$

$s_b$ 小 ⇒ 背景静止 ⇒ 转台；$s_b$ 大 ⇒ 环绕。本项目要求**判定必须在合成背景之前完成**
（检测器输入原始帧 + 掩码，采样 12 帧成对比较），否则掩码把背景涂黑后差异被人为抹平，
同一素材两次运行可能给出不同模式（实测 SSIM 0.9267 vs 0.9485）。

### 3.2 背景合成监督（全图损失分解）

只在掩码内取均值时，损失量级随前景占比漂移（实测各视角前景 12%~55% ⇒ 同误差下 loss 差 4.6 倍），
且**掩码外完全无监督**。改成先把 GT 在掩码外合成为已知背景色

$$\tilde I(u)=M(u)\,I(u)+(1-M(u))\,b,$$

再在全图算损失，等价于

$$\mathcal{L}=\frac{1}{|\Omega|}\left[\sum_{M}\ell(C,I)+\sum_{\bar M}\ell(C,b)\right],$$

其中第一项拟合前景、第二项把背景压成已知色 $b$。

第二项正是"物体外不许生成高斯"的软约束（$\ell$ 取 L1）。配套地，掩码用形态学腐蚀
$M_\epsilon = M\ominus K_\epsilon$（`mask_erode_px`）收缩，避免把掩码外沿的亮背景学成轮廓高光。

### 3.3 退火式"推实"正则（crispness）

对不透明度加对称惩罚

$$\mathcal{L}_{\text{crisp}}=\lambda\,\frac{1}{N}\sum_j \alpha_j\,(1-\alpha_j),$$

它在 $\alpha\in\lbrace 0,1\rbrace$ 处最小。其梯度

$$\frac{\partial\mathcal{L}_{\text{crisp}}}{\partial o_j}
=-\lambda\,\alpha_j(1-\alpha_j)(1-2\alpha_j)$$

在 $\alpha\to0$ 时**同时**被 $\alpha_j(1-\alpha_j)$ 压小 ⇒ 对"淡高斯云"几乎无推力；
但它对"半透明堆叠（$\alpha\approx0.5$）"有效。实测：全程施加 $\lambda=0.02$ 会把 `op>0.9` 从 15% 推到 58%，
但 SSIM −0.0043 且出现亮斑；改为**退火**（前 70% 不施加）后 `op>0.9` 只到 22%，SSIM 仅 −0.0003。
所以默认 `opacity_crisp_reg=0.02`、`opacity_crisp_warmup=0.7`。

**对称正则不够，还需要单边下限。** 只靠 crisp 项，`op<0.1` 的高斯仍占 85%。加上单边下限

$$\mathcal{L}_{\text{floor}}=\lambda_f\,\frac1N\sum_j\mathrm{relu}(f-\alpha_j),
\qquad
\frac{\partial \mathcal{L}_{\text{floor}}}{\partial o_j}
=-\lambda_f\,\alpha_j(1-\alpha_j)\,\mathbf{1}[\alpha_j\lt f],$$

（只能靠**抬高** $\alpha$ 来解除）后，在默认设置（30k、`cap_max` 600k）实测：

| 指标 | 无下限 | **下限 0.10、权重 0.5** |
|---|---|---|
| `op` 中位 | 0.031 | **0.101** |
| `op` 小于 0.1 的占比 | 85.0% | **4.0%** |
| 掩码 PSNR | 18.840 | **18.849** |
| 掩码 SSIM | 0.9654 | **0.9657** |

⇒ **几乎免费**，因此自 2026-10-26 起**默认开启**（`train.opacity_floor=0.10`、`opacity_floor_reg=0.5`）。
注意它与"容量"有耦合：`quality` 预设把 `cap_max` 压到 300k 时，同样的下限只能把 `op<0.1` 降到 61.8%。

### 3.4 深度语义统一（ED 而非Σwz）

一条视线的期望深度是**归一化**的加权平均

$$D(u)=\frac{\sum_j w_j z_j}{\sum_j w_j},\qquad w_j=\alpha_j\prod_{k\lt j}(1-\alpha_k),$$

而 gsplat 的 `render_mode="D"` 返回的是**未归一化**的 $\sum_j w_j z_j$。二者相差 $1/\sum_j w_j$ 倍。
训练侧若用后者算法线/失真，而导出侧用前者反投影，两段就在优化**不同的量**。
由 ED 求法线的切向量为

$$\mathbf{P}_u=z_u\,\mathbf{m}+z\,(1/f_x,0,0)^{T}$$

$$\mathbf{P}_v=z_v\,\mathbf{m}+z\,(0,1/f_y,0)^{T}$$

$$\mathbf{n}=\frac{\mathbf{P}_u\times\mathbf{P}_v}{|\mathbf{P}_u\times\mathbf{P}_v|}$$

其中 $\mathbf{m}=K^{-1}[u,v,1]^{T}$。用未归一化深度时，$z_u$ 被 $w$ 的梯度污染，
正对相机的平面会算出偏离 $26.6^\circ$ 的法线（实测）；用 ED 则给出 $[0,0,-1]$（偏差小于 1°，有单测）。

### 3.5 网格封闭性：一条**可证明**的判据

网格是二维复形，其边界算子 $\partial$ 作用在面集合上。**封闭**（无水密性漏洞）等价于

$$\partial_2 F=\emptyset
\iff \mathrm{card}\lbrace e:\,\deg(e)=1\rbrace=0\;\wedge\;\mathrm{card}\lbrace e:\,\deg(e)\ge3\rbrace=0,$$

$\deg(e)$ 是共享该边的面数；再要求顶点是流形（每个顶点的邻接面构成单一扇区）。
补洞就是在边界环 $C\subset\partial_2F$ 上贴一张"帽" $H$，使

$$\partial_2(F\cup H)=\emptyset$$

并且新面 $H$ 与原有面 $F$ 在共享边上的**方向相反**。

**逆向绕序**这一步是必要的：若新面与旧面在共享边上同向，则 $\partial$ 不会抵消，
Open3D 的 `is_watertight()` 仍返回 False（实测边界边已归零却仍判 False，正是朝向不一致）。
实测 img5：边界边 $3278\to0$、面数 $278{,}810\to282{,}088$、`closed=True`。

### 3.6 期望深度精修：带回溯的单调下降

把网格顶点 $x_v$ 投影到视角 $i$，比较其深度 $z_v$ 与 ED 深度 $D_i(u_v)$，沿法线做**阻尼**位移

$$\Delta x_v=\eta\cdot\frac{1}{|\mathcal{V}(v)|}\sum_{i\in\mathcal{V}(v)}
(D_i(u_v)-z_v)\cos\theta_{iv},$$

$\cos\theta_{iv}$ 抑制掠射角放大（深度误差在掠射方向被夸大 $1/\cos\theta$）。
每步之后若中位残差 $r^{(k)}=\mathrm{median}|\Delta|$ 变大，就**回退到最优顶点并把步长减半**：

$$\eta_{k+1}=\tfrac12\eta_k,\qquad x^{(k+1)}=x^{\star}$$

仅在 $r^{(k+1)}$ 大于最优残差 $r^{\star}$ 时执行这次回退。

这是 ICP 的标准 backtracking，保证 $r^{(k)}$ 单调不增。实测：不加回溯时残差 $0.0643\to0.0789$（越修越糟），
加上之后 $0.0643\to0.0551$（−14%，6 轮收敛）。

### 3.7 指标族：把"观感"拆成可计算的量

| 指标 | 判读 |
|---|---|
| 掩码内 PSNR / SSIM | 黑底模式下唯一有意义（背景是否合成正确不影响它）|
| 轮廓光晕 rim | 大于 0.10 ⇒ 可见白边 |
| 轨道彩度 chi | 大于 0.12 ⇒ 彩虹斑块 |
| 实心度（不透明度中位） | 中位小于 0.1 且多数高斯小于 0.1 ⇒ 观感发雾 |
| 掩码松紧 rho | 大于 0.5 ⇒ 掩码过松（会把背景学进模型）|
| 网格封闭 | 边界边 0、非流形 0、顶点流形（见 3.5）|

各量的定义：

$$\mathrm{PSNR}=10\log_{10}\frac{1}{\mathrm{MSE}_M}$$

$$\mathrm{rim}=\bar{L}_{ring}-\bar{L}_{inner}$$

$$\chi=\mathrm{mean}_{u}\left(\max_c C_c-\min_c C_c\right)$$

$$\rho=\frac{|M\setminus M_{gt}|}{|M_{gt}|}$$

### 3.8 实验性（本地测试中，**未纳入推荐**）：视线级不透明度下限

设每条视线的累计不透明度 $T(u)=1-\prod_j(1-\alpha_j(u))$，对其加单边下限

$$\mathcal{L}_{\text{ray}}=\frac{1}{|\mathcal{M}|}\sum_{u\in\mathcal{M}}
\mathrm{relu}(\tau-T(u)),\qquad
\frac{\partial\mathcal{L}_{\text{ray}}}{\partial o_j}
=-\alpha_j\,(1-T)\,\mathbf{1}[T\lt \tau].$$

形式上比逐高斯的 $-\alpha_j(1-\alpha_j)$ 在 $\alpha\to0$ 时强得多（$\alpha=0.01,T=0.1$ 时差约 90 倍）。

**但实测前提不成立**：本项目数据上掩码内 $T$ 的中位数已是 **0.9929**（均值 0.9649，仅 8.84% 像素 $\lt 0.9$），
于是该惩罚几乎恒为 0 —— 训练后不透明度分布与基线逐位相同。
机制保留为 `train.ray_opacity_floor` / `ray_opacity_reg` / `ray_opacity_warmup`（**默认 0**），
数学与梯度性质由 `tests/test_ray_opacity_floor.py` 固定；负结果记录见 docs/03 G160。

---

### 3.9 深度集中度正则：把"朦胧"变成可优化的量

3.8 的负结果说明"朦胧"不是累计不透明度不足（$T$ 中位 0.99），而是**同一条视线上颜色的来源分布在很多不同深度**：
表面位置不确定 ⇒ 换视角就"薄"。2DGS 用成对失真直接惩罚这种分布

$$\mathcal{L}_{d}=\sum_{u}\sum_{i,j}w_iw_j\,|z_i-z_j|,$$

但它需要光栅化核里**逐像素的高斯列表**（要自己写 CUDA）。本项目给出的可微替代是沿射线的**深度方差**：

$$\mathbb{E}_u[z]=\frac{\sum_j w_jz_j}{\sum_j w_j},\qquad
\mathrm{Var}_u[z]=\frac{\sum_j w_jz_j^{2}}{\sum_j w_j}-\left(\mathbb{E}_u[z]\right)^{2},$$

$$\mathcal{L}_{\text{var}}=\frac{1}{|\mathcal{M}|}\sum_{u\in\mathcal{M}}
\frac{\mathrm{Var}_u[z]}{s^{2}},\qquad s=\text{scene scale}.$$

**怎么算出来**（不需要自定义核）：gsplat 的 `colors` 支持任意通道，把逐高斯的相机深度 $z_j$ 与 $z_j^{2}$
当颜色通道渲染一次，就同时得到 $\sum w_jz_j$、$\sum w_jz_j^{2}$，而返回的 alpha 就是 $\sum w_j$。
一次额外的光栅化即可，反向传播由 gsplat 自动完成。

**实测（img5，8 个留出视角，15k 步，同一份数据）**：

| $\lambda_{\mathrm{var}}$ | 方差中位 | 表面厚度 | p90 方差 | 掩码 PSNR | 掩码 SSIM | LPIPS |
|---|---|---|---|---|---|---|
| 0（关）| 0.00371 | 0.0609 | 0.0297 | 29.18 | 0.9819 | **0.2401** |
| 0.3 | 0.00241（−35%）| 0.0491 | 0.0109（−63%）| **29.22** | **0.9821** | 0.2410 |
| **1.0（默认）** | **0.00150**（−60%）| **0.0387** | **0.0055**（−81%）| 29.20 | 0.9820 | 0.2408 |
| 3.0 | **0.00089**（−76%）| 0.0298 | 0.0032（−89%）| 29.15 | 0.9818 | 0.2414 |

⇒ **剂量-响应单调**：λ 越大深度越集中；而 PSNR（0.07 dB）与 LPIPS（0.0013）的变化
**全部在噪声内**，实心度（`op` 中位 0.101）完全不变。因此默认取 **λ=1.0**：方差 **−60%**、尾部 **−81%**。

这与"治朦胧"的目标一致：模型把同一像素的颜色压到更窄的深度范围，几何更像"面"而不是"云"。

**60k 验证（`solid` 预设，含导出）**：迭代越久，模型越倾向用多层淡高斯拟合外观，方差会自己涨上去 ——
这正是"越训越朦胧"的机制：

| 作业 | 配置 | 掩码 PSNR | 掩码 SSIM | LPIPS | 方差中位 | p90 |
|---|---|---|---|---|---|---|
| v4 | 无深度方差、$\lambda_{\mathrm{lpips}}=0.1$ | **30.56** | **0.9866** | **0.2353** | 0.01448 | 0.0969 |
| **solid-v2** | 深度方差 1.0、$\lambda_{\mathrm{lpips}}=0.05$ | 30.54 | 0.9861 | 0.2399 | **0.00186** | **0.0071** |
| 差值 | | −0.02 dB | −0.0005 | **+0.0046** | **−87%** | **−93%** |

⇒ 60k 下方差收益更大（**−87%**），代价是 LPIPS **+0.0046**（2% 相对）。
**"15k 上几乎免费"不等于"60k 上几乎免费"**：要不要开取决于交付目标
（要"不薄不糊"就开，只拼 LPIPS 榜单就关）。详见 `docs/03` G165。

> 配置：`train.lambda_depth_var`（默认 **1.0**，设 0 关闭）、`train.depth_var_warmup`（0.3）、
> `train.depth_var_mode`（`var` / `std` / `rvar`）。
> 与 2DGS 原式的差别：那边是 $L_1$ 型成对差、需自定义核；这里是二阶矩（$L_2$ 型），
> 两者都压"深度分散"，但不相等。实测证据见 `scripts/probe_depth_variance.py`。

### 3.10 细节梯度损失：让高斯"有质感"

颜色损失、SSIM、LPIPS 都**奖励"平均掉高频"**的解：把纹理抹平能同时降低 L1 与 LPIPS。
表现就是"形状对、表面发蜡/发糊"。梯度损失直接约束图像的一阶结构：

$$\mathcal{L}_{\mathrm{grad}}=\lambda_g\,\frac{1}{|\mathcal{M}|}\sum_{u\in\mathcal{M}}
\left|\nabla C(u)-\nabla\tilde I(u)\right|_1$$

平坦区域梯度接近 0、几乎不贡献；边缘处差一点就被惩罚。所以它用**几乎相同的预算**换更清楚的纹理，
而且不像 SSIM 那样允许"整体低频化"。

**实测（img5，15k 步，8 留出视角，其余全默认）**：

| $\lambda_g$ | 高斯数 | 掩码 PSNR | 掩码 SSIM | LPIPS | 高频能量比 | 高频相关 |
|---|---|---|---|---|---|---|
| 0（关）| 224,885 | 29.19 | 0.9819 | 0.2407 | 1.201 | 0.750 |
| 0.1 | 236,700 | 29.91 | 0.9844 | 0.2350 | 1.094 | 0.785 |
| **0.3（默认）** | 257,712 | 31.22 | 0.9879 | 0.2248 | 1.033 | 0.807 |
| **1.0（`solid` 预设）** | 325,441 | **33.94** | **0.9928** | **0.2041** | **0.989** | **0.839** |

16 视角复核同序（27.67 → 29.40 → 31.53 dB），不是视角偶然。

⇒ **这是目前收益最大的一项**：$\lambda_g=1.0$ 时 PSNR **+4.75 dB**、LPIPS **−15%**、高频能量比回到 1.0、
高频相关 0.750→0.839；代价是高斯数 **+45%**、训练时间 +26%。
实心度不受影响（`op` 中位 0.101~0.102）。视觉上屋面瓦纹更清楚、涂抹感减少。

> 配置：`train.lambda_grad`（默认 **0.3**；`solid` 预设 1.0）、`train.grad_warmup`（0.2，前 20% 不施加）。
> 指标：`scripts/probe_texture_quality.py`（梯度能量比 / 高频能量比 / 高频相关）。

## 四、安装

### 方式 A：可视化安装器（推荐）

```bat
python installer\gsflow_installer.py
```

三页向导：**环境检测**（Python / uv / Tkinter / GPU / 磁盘 / 写权限）→ **安装路径 + 组件勾选**
（core / gpu / segment / tools / weights，可勾"复用已有安装"省数 GB 流量）→ **安装**（实时日志）。
最后会自动跑 `gsflow doctor` 验收。

无人值守：

```bat
python installer\gsflow_installer.py --silent --path F:\gsflow ^
    --components core,gpu,tools,weights --reuse-from F:\3dgs-and-ai
python installer\gsflow_installer.py --check-only          :: 只检测，不安装
```

安装器**不含测试内容**（不复制 `tests/`、不预置实验产物），可安装到任意目录。

### 方式 B：脚本

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -All
scripts\gsflow.cmd doctor
```

### 硬件与约束（本机实测基线）

| 项 | 值 |
|---|---|
| GPU | RTX 5060 Laptop **8 GB**，Blackwell **sm_120**，驱动 573.22 |
| CUDA / torch | **必须 ≥12.8**，torch 用 **cu128** 轮子（≥2.7 才含 sm_120 内核）|
| 内存/磁盘 | 16 GB / 建议 ≥20 GB 空闲 |
| 实测速度 | 30k 迭代、~200 视图、1600 长边 ≈ **16 分钟**；quality 预设（60k + 每步 LPIPS）≈ **45–75 分钟** |


## 五、跑一个作业

> 这一节是**照做就能跑通**的操作流程。原理与参数见 [`docs/05-训练框架.md`](docs/05-训练框架.md)，
> 完整参数速查见本文件「五之二」。

### 5.1 CLI 全流程（推荐先跑一遍这个）

```bat
:: 0) 进入安装目录，自检一次（应全绿或只有"可选工具缺失"提示）
cd /d F:\gsflow
scripts\gsflow.cmd doctor

:: 1) 放数据：视频或照片目录，路径尽量不含中文与空格
::    F:\gsflow\data\IMG5.mp4      或      F:\gsflow\data\photos\

:: 2) 一条命令跑完五个阶段（进度实时打印，可随时 Ctrl+C 中断）
scripts\gsflow.cmd run --video data\IMG5.mp4 --job-id img5 --stages frames,mask,colmap,train,export

::    想用"画质+实心度"都更好的配置（约 60~75 分钟）：
scripts\gsflow.cmd run --video data\IMG5.mp4 --job-id img5 --preset solid

:: 3) 中途看进度（另开一个终端）
scripts\gsflow.cmd jobs
scripts\gsflow.cmd status img5

:: 4) 断了/想改参数：**只重跑某个阶段**，前面已完成的产物会复用
scripts\gsflow.cmd run --job-id img5 --stages train,export --force train,export
::    例如只重试 COLMAP：
scripts\gsflow.cmd run --job-id img5 --stages colmap --force colmap

:: 5) 看产物
dir workspace\img5\05_export

:: 6) 质量体检（一条命令跑完全部检查；训练占卡时加 --no-gpu）
env\venv\Scripts\python.exe scripts\check_quality_suite.py --job img5
```

**产物在哪**（每个作业一个目录，可整目录打包搬走）：

```
workspace\img5\
  01_images_raw\        抽出的原始帧
  02_masks\             前景掩码       02_images_clean\  掩码合成后的训练图
  03_colmap\sparse\     相机位姿       03_colmap\report.json  注册率/重投影
  04_train\point_cloud\iteration_*\point_cloud.ply   3DGS 模型（59 维，含 SH）
  04_train\metrics.json 训练指标
  05_export\            model*.glb / model.splat / model_mesh*.obj|ply|glb / viewer*.html
  logs\events.jsonl     全部事件（进度、告警、失败原因）
```

### 5.2 Web GUI 全流程（同一套流程核心，等价于上面的命令）

```bat
scripts\web.cmd            :: 启动后打开 http://127.0.0.1:8787
```

界面有六个页签，按顺序用：

| 页签 | 做什么 | 关键操作 |
|---|---|---|
| **① 新建作业** | 填输入与参数 | 「输入」填视频/照片路径 → 「作业 ID」留空则自动生成 → 「阶段」默认五个全跑（也可只勾某几个）→ 「关键参数」里选**训练预设**（`balanced` / `quality` / `solid`）→ 需要网格就把「3.5 网格导出」的 `mesh_method` 设 `tsdf`、格式加 `mesh` → 点 **开始重建** |
| **② 作业与进度** | 看实时日志 | 进度条 + 逐行事件；失败会显示原因；**取消当前作业**按钮随时停止 |
| **③ 模型查看** | 看 3D 结果 | 选作业 → 选模式（**服务端预览**=不依赖 WebGL，最可靠 / **高斯泼溅**=59 维含 SH 的真实高斯 / **点云**=3DGS 原始形态 / **网格** / **带贴图网格**）；左键拖拽旋转、右键/Shift 平移、滚轮缩放；「服务端渲染当前视角」用 gsplat 官方光栅化器出图（与对照图同一渲染器）|
| **④ 网格导出** | 从已有作业再导一次网格 | 选作业 + 方法 + 目标面数 → 开始（复用同一模型的位姿与深度，不必重训）|
| **⑤ 环境自检** | 软硬件与依赖 | 与 `gsflow doctor` 同一份报告；缺什么会给出处理建议 |
| **⑥ 设置 / API Key** | 密钥与路径 | 填 HuggingFace token（下载权重用）；保存后写入 `config/config.toml`（不入库）|

> **GUI 与 CLI 完全等价**：界面提交的参数会拍平成 `--set a.b=c`，等价命令显示在「新建作业」下方，
> 可以直接复制到终端执行或写进脚本。

### 5.3 两种常见用法

```bat
:: A. 快速试水（先确认能出东西：15k 迭代、约 8~10 分钟）
scripts\gsflow.cmd run --video data\x.mp4 --job-id try1 --set train.iterations=15000

:: B. 正式交付（solid 预设 = 60k + 每步 LPIPS + 单边不透明度下限）
scripts\gsflow.cmd run --video data\x.mp4 --job-id final --preset solid
```

**预设**（单一映射源 `src/gsflow/presets.py`，CLI 与 GUI 共用）：

| 预设 | 内容 | 适用 |
|---|---|---|
| `balanced` | 当前默认（30k、densify_until 4000、内置密度控制）| 快速验证 |
| `quality` | 60k、`cap_max` 300k、densify_until 10000、每步 LPIPS 0.05、2DGS 面元正则 0.02、推实正则 0.05（warmup 0.4）| **画质优先** |
| `solid` | 在 `quality` 基础上加**单边不透明度下限**（`opacity_floor=0.10`，reg 0.5，warmup 0.3）+ `cap_max` 400k | **推荐**：画质与实心度同时更好 |

**实测（img5，8 个留出视角，同一把论文对齐的尺子）**：

> 下表是 **2026-10-26 之前、下限正则尚未默认开启时** 的数据；今后 alanced 也带下限（实测 op<0.1 85%→4%、PSNR +0.01、SSIM +0.0003，见 3.3）。

| 方案 | `op` 小于 0.1 的占比 | 掩码 PSNR | 掩码 SSIM | LPIPS |
|---|---|---|---|---|
| `balanced`（30k）| 85.0% | 29.87 | 0.9846 | 0.2370 |
| `quality`（无 floor，cap 300k）| 100%（更淡）| 30.16 | **0.9868** | **0.2300** |
| `solid`（floor 0.10 + cap 400k）| **59.9%** | **30.56** | 0.9866 | 0.2353 |

⇒ **联合标定可以同时改善两侧**：`solid` 比 `balanced` 的掩码 PSNR 高 **0.69 dB**，实心度中位高 **3.1 倍**。
关键是"**容量足够大以保住细节 + 下限正则防止塌成雾**"两者同时给：
只给下限但容量小会让画质回落（旧版 v3：cap 300k → PSNR 29.86）；只给大容量不给下限（`quality`）则 100% 淡高斯。
另注意：`op*(1-op)` 这种**对称**正则实测**完全无效**（op→0 时惩罚与梯度一起趋零），必须用单边下限。

> `train.densify_strategy=official`（gsplat 官方 `DefaultStrategy`）**目前不推荐**：
> 实测在 step 10000 涨到 251 万高斯（我们已补上漏传的 `cap_max`，但阈值仍需重新标定）。

---

### 5.4 流程逻辑（数学）：每一步的公式与默认值

> 这一节把五个阶段的**核心公式 + 代码里的默认值**逐条列出，便于核对实现、也便于改参数时知道自己在动哪个量。
> 符号：内参 $K$，外参 $(R_i,t_i)$，像素 $u$，深度 $z$，高斯 $\lbrace(\mu_j,\Sigma_j,\alpha_j,c_j)\rbrace$，
> 不透明度 $\alpha_j=\sigma(o_j)$，透射率 $T_j=\prod_{k\lt j}(1-\alpha_k)$。

#### ① 抽帧（frames）

帧率采样：在时间轴上等间隔取 $t_k=t_0+k/f$，$f=$ `frames.fps`（默认 **2.0**），上限 `frames.count`（120）。
清晰度筛选：$\;s(I)=\mathrm{Var}(\nabla^2 I)$，$s\lt $`frames.blur_threshold` 的帧丢弃（默认 **0 = 不过滤**）。
分辨率：长边缩放到 `frames.long_side`（默认 **1920**）。
自动裁边：对序列求逐像素时间方差 $\mathrm{Var}_t I_t(u)$，保留高方差行/列的包围盒
（`frames.auto_crop` 逻辑见 3.1；阈值 `auto_crop_threshold` 默认 **0.75**）。

#### ② 掩码与背景清理（mask）

分割给出 $M(u)\in\lbrace 0,1\rbrace$（`mask.backend`：`rembg` / `fastsam` / `classical`），
训练图按 $\;I'(u)=M(u)I(u)+(1-M(u))b\;$ 合成，$b$ 由 `mask.mode` 决定：
`black` ⇒ $b=0$（默认），`keep` ⇒ 保留原背景（转台拍摄必须用这个，见 3.1）。

形态学与门限（默认值）：`mask.dilate_px=4`、`mask.feather_px=2`、`mask.erode_px=0`
（$M_\epsilon=M\ominus K_\epsilon$）、`mask.min_area_ratio=0.002`、`mask.keep_largest=true`；
退化保护：前景占比落在 $[2\%,\,90\%]$ 之外视为失败，按 `mask.on_failure`（默认 `keep`）降级。

#### ③ COLMAP 稀疏重建（colmap）

相机模型 `colmap.cameras="SIMPLE_RADIAL"`（$f,c_x,c_y,k_1$，`single_camera=true`）：

$$\pi(X)=K[\tfrac{x}{z},\tfrac{y}{z},1]^\top,\quad (x,y,z)=R_iX+t_i,\qquad
r\mapsto r\,(1+k_1 r^2).$$

匹配：`colmap.matcher="auto"` —— 视频/顺序拍摄走 **sequential**（`sequential_overlap=12`），
无序照片走 **exhaustive**；特征上限 `max_num_features=8192`、`max_num_matches=32768`、
`min_num_matches=32`，图像长边 `max_image_size=2048`。
BA 目标与质量门限：

$$\min_{\lbrace R_i,t_i,X_j\rbrace}\sum_{(i,j)\in\mathcal{O}}|\tilde x_{ij}-\pi(R_iX_j+t_i)|^2$$

质量门限：**注册率**不低于 0.75（`quality_min_registered_ratio`），
**平均重投影误差** $\bar{e}$ 不超过 2.0 px（`quality_max_reproj_error`）。

#### ④ 3DGS 训练（train）

**前向**（$\alpha$-合成，按深度排序）：

$$C(u)=\sum_{j}\alpha_jT_jc_j,\qquad
D(u)=\frac{\sum_j \alpha_jT_jz_j}{\sum_j\alpha_jT_j},\qquad
T_j=\prod_{k\lt j}(1-\alpha_k)$$

其中 $D(u)$ 是**期望深度（ED）**，训练与导出统一用它。

**损失组成**（默认值）：

$$\mathcal{L}=\mathcal{L}_{color}+\lambda_{sh}\mathcal{L}_{sh}+\lambda_{lpips}\mathcal{L}_{LPIPS}+\lambda_{n}\mathcal{L}_{normal}+\lambda_{d}\mathcal{L}_{dist}+\mathcal{L}_{floor}+\mathcal{L}_{bg}$$

其中各项：

$$\mathcal{L}_{color}=(1-\lambda_s)\left|C-\tilde I\right|_1+\lambda_s\left(1-\mathrm{SSIM}\right)$$

$$\mathcal{L}_{sh}=\left|\theta_{rest}\right|_2^2$$

$$\mathcal{L}_{floor}=\lambda_f\frac{1}{N}\sum_j\mathrm{relu}(f-\alpha_j)$$

$$\mathcal{L}_{bg}=\lambda_{bg}\frac{1}{\left|\bar M\right|}\sum_{\bar M}\alpha$$

| 项 | 默认 | 说明 |
|---|---|---|
| `ssim_weight`（$\lambda_s$）| 0.2 | 结构相似度权重 |
| $\lambda_{sh}$（`sh_rest_reg`）| 0.003 | 抑制斜视彩虹（只压高阶 SH）|
| `lambda_lpips` / `lpips_every` | 0 / 1 | 开启时必须**每步**算（G152）|
| `lambda_normal` / `lambda_depth_distortion` | 0 / 0 | 2DGS 面元约束（在 ED 深度上）|
| `lambda_depth_var` / `depth_var_warmup` | **1.0** / 0.3 | 深度方差正则（治"朦胧"），实测 −60% 方差、−81% 尾部且几乎不损画质（3.9）|
| `opacity_floor` / `opacity_floor_reg` | **0.10 / 0.5** | 单边下限，实测几乎免费（3.3）|
| `bg_alpha_reg`（$\lambda_{bg}$）| 0.1 | 掩码外 alpha 抑制 |

**自适应密度控制**（`densify_*`，默认）：从 `densify_from=500` 开始、到 `densify_until=4000` 结束，
每 `densify_interval=100` 步一次；判据为屏幕空间梯度

$$g_j=\left|\frac{\partial\mathcal{L}}{\partial \mu_j^{2D}}\right|$$

当 $g_j$ 大于阈值 $2\times10^{-4}$（`densify_grad2d_threshold`）时克隆或分裂该高斯。

**剪枝**：$\alpha_j\lt \varepsilon_\alpha$（`prune_opacity=0.005`）或
$\max_k s_{jk}\gt \rho_s\cdot\text{scene-scale}$（`prune_scale3d=0.1`）删掉；
另有 `max_scale_ratio=0.05`（每步尺度上限）、`max_aspect=50`（长宽比上限）、`cap_max=600000`。

**不透明度重置**：每 `opacity_reset_interval` 步把 $\sigma(o_j)\leftarrow\min(\sigma(o_j),\rho)$，
$\rho=2\varepsilon_\alpha=0.01$（官方行为；`revised_opacity=0.1` 是分裂系数，不是重置值）。
本项目还提供 `auto_extend_densify_when_reset`（默认 true）：重置后把密度窗口顺延，避免"重置即停密"。

**学习率**（`lr_*`，默认）：`lr_means_scene_scale=false` ⇒ $\eta_\mu=1.6\times10^{-4}$ 固定；
$\eta_{\text{sh DC}}=2.5\times10^{-3}$、$\eta_{\text{sh rest}}$=`lr_shs=0.0025`、$\eta_\alpha=0.025$、
$\eta_{\text{scale}}=5\times10^{-3}$、$\eta_{\text{quat}}=10^{-3}$；`lr_schedule="cosine"`，
$\eta(t)=\eta_{\min}+(\eta_0-\eta_{\min})\cdot\frac12(1+\cos\frac{\pi t}{T})$，$\eta_{\min}=$\texttt{lr-min-ratio}$\eta_0$（0.1）。

#### ⑤ 导出（export）

**ED 深度渲染**（与训练同一套 ED）：`export.mesh_width=640`、`export.mesh_max_views=80`；
未归一化的 $\sum w z$ 不参与反投影（3.4）。

**TSDF 融合**（`mesh_method="tsdf"`，本项目推荐给薄结构）：

$$f(v)=\frac{\sum_i w_i(v)\,d_i(v)}{\sum_i w_i(v)},\qquad
d_i(v)=\mathrm{clamp}(D_i(u_v)-z_v,\,-\delta,\delta),$$

体素 $\delta=$`scene_scale`$/$`mesh_voxel_div`（默认 400 ⇒ 实测 0.008~0.016），
深度过滤保留率实测 98.9~99.1%，零等值面用 Marching Cubes 提取。

**泊松**（`mesh_method="poisson"|"2dgs"`）：解 $\Delta\chi=\nabla\cdot\mathbf{V}$，
$\mathbf{V}=\sum_j n_j\delta_{\mu_j}$，等值面 $\chi=\tfrac12$；
默认 `mesh_poisson_depth=8`（实测 8 比 9 的相邻面法线翻转率低 10 倍）。

**后处理**：低密度裁剪 `mesh_density_quantile=0.001`、裁剪分位 `mesh_crop_quantile=1.0`、
预简化 `mesh_presimplify_ratio=10`、Taubin 平滑 `mesh_smooth_iters=5`、**QEM** 到
`mesh_target_triangles=400000`：

$$\Delta(v)=\sum_{p\in\mathcal{P}(v)}\left(n_p^{T}(\bar v-p)\right)^{2}$$

其中 $Q$ 是各平面二次型 $K_p$ 之和，$K_p$ 由平面 $(n_p,p)$ 构造。

> 这里刻意用**平面距离平方**的等价写法，而不是 $2\times2$ 分块矩阵：矩阵里的换行 `\\` 会被
> Markdown 当成转义吃掉（变成单个 `\`），矩阵会塌成一行 —— 这是 GitHub 上常见的一类渲染事故。

**水密化**（`mesh_watertight=true`）：补洞使 $\partial_2(F\cup H)=\emptyset$ 且新面与旧面**反向共享边**（3.5）。

**贴图**（`mesh_texture=false` 默认关闭）：xatlas 展开最小化

$$E_{\text{stretch}}=\sum_t A_t(| J_t|_F^2+| J_t^{-1}|_F^2),$$

再按可见性加权采样 $T(x)=\sum_i w_i(x)I_i(\pi_i(x))/\sum_i w_i(x)$，
`mesh_texture_size=2048`（实测 UV 覆盖 100%、黑 texel 0.00%）。

## 五之二、参数速查（关键项）

> 完整默认值见 src/gsflow/config.py:DEFAULTS；这里只列**直接影响质量**的项。
> CLI 用 --set section.key=value 覆盖，也可写进 config/config.toml。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `export.mesh_method` | 2dgs | 网格路线：poisson / 2dgs / tsdf |
| `export.mesh_width` | 640 | 渲染深度分辨率 |
| `export.mesh_max_views` | 80 | 参与融合的视图数 |
| `export.mesh_poisson_depth` | 8 | 泊松八叉树层数 |
| `export.mesh_density_quantile` | 0.001 | 低密度包络面剪除分位 |
| `export.mesh_crop_quantile` | 1.0 | 网格聚焦的分位数裁剪（1.0=不裁） |
| `export.mesh_trim_sigma` | 2.5 | 边界剪切宽度（σ） |
| `export.mesh_consistency_min_views` | 2 | 多视核查最少一致视图数 |
| `export.mesh_presimplify_ratio` | 10.0 | 预简化比例（为平滑降规模） |
| `export.mesh_target_triangles` | 400000 | 目标面数（QEM） |
| `export.mesh_smooth_iters` | 5 | Taubin 平滑迭代 |
| `export.mesh_texture` | false | 是否烘焙贴图（可选，耗时） |
| `train.iterations` | 30000 | 迭代数 |
| `train.resolution` | 1600 | 训练长边上限 |
| `train.cap_max` | 600000 | 高斯数上限 |
| `train.densify_until` | 4000 | 致密化截止步数 |
| `train.max_scale_ratio` | 0.05 | 每步尺度上限（防针状高斯） |
| `train.bg_alpha_reg` | 0.1 | 背景 alpha 正则权重 |
| `train.max_aspect` | 50.0 | 最大长宽比（防扁平高斯） |
| `train.lambda_lpips` | 0.0 | 感知损失权重（需每步算） |
| `train.lambda_depth_distortion` | 0.0 | 2DGS 深度失真正则权重 |
| `train.rasterize_mode` | classic | 光栅化：classic / antialiased |
| `train.densify_criterion` | grad2d | 致密判据：grad2d / grad3d |
| `train.densify_grad2d_threshold` | 0.0002 | grad2d 阈值 |
| `train.absgrad` | false | AbsGS 绝对梯度（仅 official 策略） |
| `train.auto_crop` | true | 帧阶段的时间方差自动裁剪 |
| `train.auto_crop_in_train` | false | 训练内自动裁剪（黑边条） |
| `frames.crop` |  | 手动裁剪框 WxH+X+Y |
| `mask.mode` | black | black=背景涂黑 / keep=保留背景 |
| `capture.mode` | auto | auto / turntable / orbit |
| `export.formats` | [glb_points, mesh, viewer_html, splat] | 导出格式列表 |
| `export.mesh_drop_unseen` | true | 丢弃从未被看到的面 |
| `export.mesh_min_views` | 1 | 面被看到的最少视图数 |
| `export.mesh_texture_reject_black` | true | 烘焙时拒绝纯黑采样 |

> **为什么看掩码内指标**：psnr_masked / ssim_masked 只在掩码内计算，是黑底模式下唯一有意义的画质指标；全图 PSNR 会把“背景是否合成正确”混进来。

## 六、数据怎么拍（决定上限的第一因素）

| 要点 | 原因（实测） |
|---|---|
| **帧数 100–400**，覆盖 360° | 少于 60 帧几何不稳；超过 400 帧收益递减但耗时线性上升 |
| **混合视角：平视 + 俯视 + 仰视** | 只有平视 ⇒ 屋顶/顶面几乎无观测，网格会塌成薄片（img4 俯视是空的，img5 补了俯视就有了屋顶）|
| 长边 ≥1280，物体占画面 ≥1/3 | 分辨率直接决定细节上限（img3 物体只有 380×250 px，GSD 3.7 mm/px）|
| 背景与物体**有对比** | 纯黑背景 + 深色物体会让掩码边界不稳 |
| 避免强烈反光/透明物体 | 3DGS 对镜面/玻璃的重建本质不可靠 |
| 固定焦距、避免数字变焦 | 内参漂移会让 COLMAP 与训练都崩 |

---

## 七、质量体检工具（本项目最实用的部分）

一条命令跑全部：

```bat
scripts\gsflow.cmd export --job img5 --json     :: 产物清单
env\venv\Scripts\python.exe scripts\check_quality_suite.py --job img5          :: 全部体检（汇总 JSON）
env\venv\Scripts\python.exe scripts\check_quality_suite.py --job img5 --no-gpu :: 训练占卡时用
```

| 工具 | 查什么 | 判读 |
|---|---|---|
| `check_mesh_quality.py` | 边界边比例、非流形、**封闭**、自交、面积/凸包 | 封闭=True、边界边=0、非流形=0；自交单独看（泊松固有）|
| `check_gaussian_solidity.py` | 不透明度中位、`op` 小于 0.1 的占比、淡体积份额 | 中位小于 0.1 且六成以上小于 0.1 ⇒ 会"朦胧/透明" |
| `probe_lpips_new.py` | 用**论文对齐**的 LPIPS 复评多个作业 | 同一把尺子才能横比；旧值不可比 |
| `check_rim_brightness.py` | 轮廓 2px 环亮度 − 内部亮度 | 大于 0.10 = 可见白边 |
| `render_orbit_sheet.py` | 3 仰角 × 6 方位出图 + 彩度 | 彩度大于 0.12 = 彩虹斑块 |
| `render_textured_mesh.py` | **带贴图网格**离屏渲染 | 看得见"空壳/飞边"；面数超过 6 万会自动简化（渲染器上限）|
| `check_mask_tightness.py` | `rim_ratio`（掩码松紧）| 大于 0.5 = 掩码太松 |
| `final_acceptance.py` | 交付验收清单 | 逐项 PASS/FAIL/SKIP（可选产物按配置跳过）|

> **为什么需要这么一套**：掩码内 PSNR 26.5 dB 的模型看起来可能仍然"朦胧"——
> 图像指标对"是否实心、是否封闭"完全不敏感。上面每个工具都是被具体问题逼出来的。

---

## 八、架构

```
输入（视频/图片）
   │
   ├─ stages/frames.py     抽帧、裁黑边、清晰度筛选
   ├─ stages/mask.py       分割 + 拍摄模式判定 + 掩码闸门（2%~90% 前景、回读校验）
   ├─ stages/colmap.py     COLMAP（顺序/穷举）→ pycolmap 回退；质量体检
   ├─ stages/train.py      gsplat 训练（致密化 / 重置 / 剪枝 / 损失组合）
   ├─ stages/export.py     点云/网格/贴图/查看器 + 水密化 + ED 精修
   └─ stages/metrics.py    汇总报告
         │
         ├─ pipeline/orchestrator.py  阶段调度、断点续跑、事件流（events.jsonl）
         ├─ cli.py                    `gsflow run/doctor/jobs/...`
         └─ web/app.py                FastAPI + 静态前端（六个页签）
```

**扩展点**：分割后端（实现 `segment/base.py:Segmenter`）、训练后端（`trainers/base.py` 协议）、
外部工具（`tools/locate.py` 适配器）。

**目录约定**（每个作业一个目录，可整目录打包迁移）：

```
workspace/作业目录/
  01_images_raw/   02_masks/   02_images_clean/
  03_colmap/       sparse/  report.json
  04_train/        data/  point_cloud/iteration_*/point_cloud.ply  metrics.json
  05_export/       model*.glb  model.splat  viewer*.html  model_mesh*.obj/ply/glb
  job.json         logs/events.jsonl
```

---

## 九、迭代路径（为什么现在是这个样子）

这个项目是**用实测把每个"看起来对"的做法推翻**长出来的，完整记录在
[`docs/03-踩坑记录.md`](docs/03-踩坑记录.md)（150+ 条，每条都有"现象 → 根因 → 修法 → 证据"）。
几个决定性的转折：

| 阶段 | 发现 | 影响 |
|---|---|---|
| 早期 | 全图 PSNR 3.4 dB 却是"好模型" | 改用**掩码内**指标 |
| 掩码 | 浮点掩码被存成全黑、bool 掩码被 `>127` 清零 | 三处 dtype bug，加 dtype 分支 + 回读校验 |
| 掩码 | 同一素材两次跑出不同拍摄模式 | 拍摄模式判定改为**确定性** |
| 检查点 | `iteration_5000` 排在 `iteration_30000` 之后 | 词法排序坑，27 处受影响 |
| COLMAP | 3.12 匹配器在空库上也失败 | 换 **4.2.1** + pycolmap 回退 |
| 训练 | "淡高斯云"（中位 0.066、67% 小于 0.1）| 退火推实正则；剪枝会掉 7.5 dB |
| 感知 | LPIPS 稀疏施加 → 反而更差 | 必须**每步**，且用**论文线性权重** |
| 网格 | TSDF 网格全黑 = 渲染器面数上限，不是网格坏 | `render_textured_mesh` 自动简化 |
| 网格 | 边界边 2% 却不封闭 | 补洞 + 自定义封闭判据 |
| 发布 | 安装器干净目录实测暴露"无 torch 就 import 崩" | 装饰器移入函数体 |

---

## 十、已知限制（不粉饰）

* **薄壳结构**（屋檐、瓦片、栏杆）在网格化时仍会被并成一片；本项目的办法是水密化 + ED 精修，
  不能凭空补出**从未被拍到**的结构（缺俯视就是缺）。
* **8 GB 显存**限制：quality 预设 + 600k 高斯在边缘（显存 4–7 GB），更大场景需降 `resolution`。
* **自交**：泊松/TSDF 结果几乎必然自交，`closed=True` 只保证拓扑封闭，不保证无自交。
* **瞬时光照**：3DGS 假设静态光照，反光/阴影变化会被烘成"视图相关的糊"。
* 官方 `DefaultStrategy` 目前需要重新标定 `grow_grad2d` 与 `cap_max` 才建议使用。

---

## 十一、测试与闸门

```bat
env\venv\Scripts\python.exe -m pytest tests -q            :: 218 passed
env\venv\Scripts\python.exe scripts\check_docs_consistency.py   :: 5/5（文档与代码一致）
env\venv\Scripts\python.exe scripts\make_tree.py                :: 重新生成 docs/04-文件树.md
env\venv\Scripts\python.exe scripts\final_acceptance.py --job img5
```

---

## 十二、许可

[MIT](LICENSE)。第三方组件：gsplat（Apache-2.0）、COLMAP（BSD）、FFmpeg（LGPL/GPL）、
gaussian-splats-3d（MIT）、three.js（MIT）、rembg / ultralytics（各自许可）。
