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
6. **负结果也写进文档**：LPIPS 每 16 步才算是**错的**（梯度与 L1/SSIM 打架，实测 LPIPS 反而变差），
   必须每步算；opacity_init=0.5 无效（淡高斯是目标函数的强吸引子）。
7. **视线级不透明度下限（本项目的不同与创新验证，见下方相关数学，该方法还在本地测试阶段）**：不逐个高斯罚 $\alpha$，
   而是直接约束**每条视线**的累计透射率 $T(u)=1-\prod_j(1-\alpha_j)$；
   梯度是 $-\alpha_j(1-T)$，比逐高斯的 $-\alpha_j(1-\alpha_j)$ 在 $\alpha\to0$ 时**强两个数量级**。

### 7.1 为什么常规重建会"颜色对、看着透"

常规 3DGS/NeRF 只用**颜色**损失。它不约束"每条视线到底有多不透明"，于是存在一族
"颜色对得上但几何是空心的"解 —— 本项目的实测数据就是证据：掩码内 PSNR 26.5~30.6 dB 的模型，
不透明度中位数只有 0.012~0.095，**100% 的高斯低于 0.1**，换到未见视角就"透"。

设第 $j$ 个高斯的不透明度 $\alpha_j=\sigma(o_j)$，一条视线穿过 $N$ 个高斯后的**累计不透明度**

$$T(u) = 1-\prod_{j=1}^{N}\,(1-\alpha_j(u)),$$

它正是光栅化器逐像素返回的 alpha。我们直接对它加**单边下限**

$$ \mathcal{L}_{\text{ray}} = \frac{1}{|\mathcal{M}|}\sum_{u\in\mathcal{M}} \mathop{\mathrm{relu}}\big(\tau - T(u)\big), \quad \tau\approx 0.9, $$

$\mathcal{M}$ 为掩码内的像素集合（背景不该被推实，否则会造出"雾墙"）。

### 7.2 梯度条件：为什么它比"逐高斯下限"强得多

对 $\alpha_j$ 求导：

$$\frac{\partial T}{\partial \alpha_j}=\prod_{k\neq j}(1-\alpha_k)
=\frac{1-T}{1-\alpha_j},\qquad
\frac{\partial \mathcal{L}_{\text{ray}}}{\partial o_j}
=-\,\alpha_j\,(1-T)\,\mathbb{1}[T<\tau].$$

对比**逐高斯**下限 $$ \mathop{\mathrm{relu}}(f-\alpha_j) $$：

$$ \frac{\partial}{\partial o_j}\mathop{\mathrm{relu}}(f-\alpha_j) = -,\alpha_j(1-\alpha_j),\mathbb{1}\{\alpha_j<f\}. $$


$f=0.1,\ \alpha_j=0.01$ 时后者约为 $9.9\times10^{-3}$；而视线级在 $T=0.1$ 时是 $0.9$ —— **相差约 90 倍**，
且差值随"还缺多少不透明度"$(1-T)$ 单调增大。换句话说：**越透的像素被推得越狠**，
而逐高斯的梯度在 $\alpha\to0$ 时反而**趋近于 0**（正是要修的状态）。
这解释了实测：把逐高斯正则权重从 $0.02$ 提到 $0.05$，分布**纹丝不动**；换成视线级约束后才出现可测的移动。

### 7.3 与"后处理剪枝"的区别

剪掉淡高斯会连带删掉承载画质的层（实测训练视角 PSNR 掉 **7.5 dB**）；
视线级约束是在**优化过程中**把不透明度重新分配到"被看见的射线"上，因此不牺牲颜色精度。

> 实现与单测：`train.ray_opacity_floor / ray_opacity_reg / ray_opacity_warmup`；
> 数学与梯度性质由 `tests/test_ray_opacity_floor.py` 固定（含"梯度非零"与"$1-T$ 条件数优势"两条）。

---

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

---

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
| `quality` | 60k、`cap_max` 300k、densify_until 10000、每步 LPIPS 0.1、2DGS 面元正则 0.02、推实正则 0.05（warmup 0.4）| **画质优先** |
| `solid` | 在 `quality` 基础上加**单边不透明度下限**（`opacity_floor=0.10`，reg 0.5，warmup 0.3）+ `cap_max` 400k | **推荐**：画质与实心度同时更好 |

**实测（img5，8 个留出视角，同一把论文对齐的尺子）**：

| 方案 | `op<0.1` 占比 | 掩码 PSNR | 掩码 SSIM | LPIPS |
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

### 5.4 流程里的数学：每一步到底在算什么

> 这一节把五个阶段的**核心公式**列出来，便于核对实现、也便于改参数时知道自己在动哪个量。
> 符号：相机内参 $K$，外参 $(R_i,t_i)$，像素 $u$，深度 $z$，高斯集合 $\{(\mu_j,\Sigma_j,\alpha_j,c_j)\}$。

**① 抽帧**：按时间等间隔采样 $t_k = k/f$（$f$ 为 `frames.fps`），并对每帧算清晰度

$$s(I)=\operatorname{Var}\big(\nabla^2 I\big),$$

低于 `blur_threshold` 的帧丢弃；需要自动裁边时按行/列的**时间方差**找高方差区（即"真正在动的区域"）。

**② 掩码与背景清理**：分割给出前景指示 $M(u)\in\{0,1\}$，训练图按

$$I'(u)=M(u)\,I(u)+\big(1-M(u)\big)\,b,\qquad b=\text{背景色（黑）}$$

合成。掩码质量用 IoU 与掩码松紧比衡量：$\text{rim\_ratio}=\dfrac{|M\setminus M_{\text{gt}}|}{|M_{\text{gt}}|}$。

**③ COLMAP 稀疏重建**：相机模型 `SIMPLE_RADIAL`（$f,c_x,c_y,k_1$），投影

$$\pi(X)=K\,\big[x/z,\;y/z,\;1\big]^\top,\qquad
K=\begin{bmatrix} f & 0 & c_x\\ 0 & f & c_y\\ 0&0&1\end{bmatrix},\; (x,y,z)=R_iX+t_i,$$

并在归一化坐标上加径向畸变 $r\mapsto r(1+k_1r^2)$。BA 最小化重投影误差

$$\min_{\{R_i,t_i,X_j\}}\;\sum_{(i,j)\in\mathcal{O}}\big\|\,\tilde{x}_{ij}-\pi(R_iX_j+t_i)\,\big\|^2,$$

报告里的 **0.45~1.08 px** 就是这个量的均值；**注册率** $=|\{i:\text{成功估计位姿}\}|/N$。

**④ 3DGS 训练**：光栅化是沿视线的 $\alpha$-合成（按深度排序，$T_j$ 为透射率）

$$C(u)=\sum_{j\in\mathcal{N}(u)}\alpha_j\,T_j\,c_j,\qquad
T_j=\prod_{k<j}\big(1-\alpha_k\big),\qquad
D(u)=\frac{\sum_j \alpha_j T_j z_j}{\sum_j \alpha_j T_j}\;,$$

其中 $D(u)$ 是**期望深度 ED**（本项目训练与导出统一使用的量；`render_mode="RGB+ED"`）。
优化目标

$$\mathcal{L}=(1-\lambda)\,\|C-\tilde{I}\|_1+\lambda\big(1-\mathrm{SSIM}(C,\tilde{I})\big)
+\sum_k w_k\,\mathcal{R}_k,$$

$\mathcal{R}_k$ 包括：SH 幅值正则 $\|\theta_{\text{rest}}\|_2^2$、不透明度正则、法线一致性、
深度失真、以及本项目的**视线级不透明度下限** $\operatorname{relu}(\tau-T)$（见第三节 7）。

自适应密度控制：对投影位置梯度大的高斯做克隆/分裂
$\big(\|\nabla_{\mu_{2D}}\mathcal{L}\|>g_{\text{th}}\big)$，对
$\alpha<\varepsilon$ 或尺度 $>s_{\max}$ 的做剪枝；每 $K$ 步把不透明度重置为
$\sigma(o)\leftarrow\min\big(\sigma(o),\rho\big)$（本项目 $\rho=0.01$，用于清除"浮点雾"）。

**⑤ 导出**：

* **TSDF 融合**：每个体素维护截断有符号距离 $f$ 与权重 $w$，

$$f(v)\leftarrow\frac{\sum_i w_i(v)\,d_i(v)}{\sum_i w_i(v)},\qquad
d_i(v)=\operatorname{clamp}\big(D_i(u_v)-z_v,\;-\delta,\;\delta\big),$$

再对 $f(v)=0$ 做 **Marching Cubes** 提取等值面（`mesh_method=tsdf`）。
* **泊松重建**（`poisson`/`2dgs` 路线）：解

$$\Delta\chi=\nabla\cdot \mathbf{V},\qquad \mathbf{V}=\sum_j n_j\,\delta_{\mu_j},$$

$\chi$ 是指示函数、$n_j$ 为定向法线；等值面 $\chi=\tfrac12$ 即网格。
* **QEM 简化**：边收缩代价

$$\Delta(v)=\bar v^\top Q\,\bar v,\qquad Q=\sum_{p\in\mathcal{P}(v)}K_p,\quad
K_p=\begin{bmatrix}n n^\top & -n\,n^\top p\\ -p^\top n n^\top & p^\top n n^\top p\end{bmatrix},$$

按 $\Delta$ 从小到大收缩直到目标面数（`mesh_target_triangles`）。
* **UV 展开**（xatlas）：最小化拉伸能量

$$E_{\text{stretch}}=\sum_{t}A_t\Big(\|J_t\|_F^2+\|J_t^{-1}\|_F^2\Big),$$

再按可见性从多视角采样颜色得到贴图：$\;T(x)=\sum_i w_i(x)\,I_i\big(\pi_i(x)\big)/\sum_i w_i(x)$，
$w_i$ 为可见性×角度权重。
* **精修**（本项目）：沿顶点法线做位移

$$\Delta x_v=\eta\cdot\frac{1}{|\mathcal{V}(v)|}\sum_{i\in\mathcal{V}(v)}
\Big(D_i(u_v)-z_v\Big)\cos\theta_{iv},$$

$\mathcal{V}(v)$ 为看到该顶点的视角集合、$\theta$ 为视线与法线夹角（抑制掠射角放大），
并用回溯式阻尼保证残差单调不增（见 `gsdata/mesh_refine.py`）。

**质量指标**（都按同一口径）：

$$\mathrm{PSNR}=10\log_{10}\frac{1}{\mathrm{MSE}_{M}},\qquad
\mathrm{SSIM}=\frac{(2\mu_a\mu_b+c_1)(2\sigma_{ab}+c_2)}
{(\mu_a^2+\mu_b^2+c_1)(\sigma_a^2+\sigma_b^2+c_2)},$$

下标 $M$ 表示**只在掩码内**统计 —— 黑底模式下全图指标会被"背景是否合成正确"污染。

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
| `check_gaussian_solidity.py` | 不透明度中位、`op<0.1` 占比、淡体积份额 | 中位 <0.1 且 `op<0.1`>60% ⇒ 会"朦胧/透明" |
| `probe_lpips_new.py` | 用**论文对齐**的 LPIPS 复评多个作业 | 同一把尺子才能横比；旧值不可比 |
| `check_rim_brightness.py` | 轮廓 2px 环亮度 − 内部亮度 | >0.10 = 可见白边 |
| `render_orbit_sheet.py` | 3 仰角 × 6 方位出图 + 彩度 | 彩度 >0.12 = 彩虹斑块 |
| `render_textured_mesh.py` | **带贴图网格**离屏渲染 | 看得见"空壳/飞边"；面数 >6 万会自动简化（渲染器上限）|
| `check_mask_tightness.py` | `rim_ratio`（掩码松紧）| >0.5 = 掩码太松 |
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
workspace/<job>/
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
| 训练 | "淡高斯云"（中位 0.066、67% <0.1）| 退火推实正则；剪枝会掉 7.5 dB |
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
