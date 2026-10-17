# gsflow · AI 图像清理 + 3DGS 三维重建全流程

> Windows 端工程化实现：**AI 先清理，3DGS 后重建**
> 原始图像/视频 → AI 前景分割与背景清理 → COLMAP 位姿恢复 → 3DGS 训练 → 导出查看
>
> 架构：**一套流程核心 + 两个入口（CLI / Web GUI）**，行为完全一致、可断点续跑。

---

## 一、当前状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| ① 抽帧 / 图片导入 | ✅ 已实测 | ffmpeg 自带，支持 fps / 按张数 / 清晰度剔除 |
| ② AI 前景分割 / 背景清理 | ✅ 已实测 | 后端可插拔；合成图实测 IoU：rembg 0.977、classical 0.93~0.98 |
| ③ COLMAP 稀疏重建 | ✅ 已实测 | 自带 COLMAP 3.12（CUDA）；自动匹配模式 + mask 接入 + 质量体检 |
| ④ 3DGS 训练（gsplat） | ✅ 已实测 | 合成基准上 **PSNR 32.90 dB（留出视图）**；免管理员便携 CUDA 编译内核通过 |
| ⑤ 导出（点云 GLB / 查看器 / 预览图） | ✅ 已实测 | GLB 通过 glTF 2.0 结构校验；预览为 gsplat **真实渲染**（渲染｜真值并排） |
| ⑥ **三角网格导出** | ✅ 已实测 | **深度反投影 → 泊松 → 平滑 → QEM 简化**；轮廓 IoU **0.66**；出 obj/ply/glb |
| ⑦ **纹理烘焙** | ✅ 已实测 | xatlas UV 展开 + 多视角采样烘漫反射贴图；UV 图集覆盖 **86%**；出带贴图 glb |
| ⑧ 质量报告 | ✅ 已实测 | 注册率、平均重投影误差、掩码占比、清晰度、PSNR、改进建议 |

**合成基准上的实测结果**（36 视图环绕一块铺满纹理的平面）：

| 指标 | 数值 |
|---|---|
| COLMAP 注册率 | 36/36（100%） |
| 平均重投影误差 | 1.08 px |
| 训练 PSNR（**留出视图**，3 张） | **32.90 dB** |
| 高斯数量 | 116,557 |
| 耗时 | 10.4 分钟（30000 迭代、700px、RTX 5060 8GB） |

> 注意：合成基准只能验证**流程是否正确**，用它评判真实场景的重建质量并不合适
> （它是一块平面，几何本身很简单）。真实质量请用实拍数据。

**真实数据上的网格质量**（122 视图环绕一个碗，`mask_loss_until=1.0`）：

| 指标 | 数值 |
|---|---|
| 网格提取路线 | 深度反投影（75 视角 @640px）→ 泊松 depth=**8** → Taubin 平滑 → QEM<br/>（原记 depth=9；实测 8 的相邻面法线翻转率低 10 倍，见下方参数表与 `docs/03` G82） |
| 交付产物（`final-v1` 实测） | 点云 ply / 实体模型 ply / 点云 glb（POINTS）/ 网格 glb（TRIANGLES）/ 带纹理 glb（TEXCOORD_0 + 2048² PNG），5/5 齐全 |
| 反投影点云 / 下采样后 | 2,579,017 → 2,311,406 点 |
| 最终面数 | 385,170（QEM 目标 40 万） |
| **轮廓 IoU**（对比高斯渲染轮廓） | **0.66** |
| 深度相对误差（中位数 / 均值 / p90） | 2.39% / 3.83% / 9.13% |
| 纹理烘焙 | UV 图集覆盖 86.5%，1024² 贴图 |

> 该模型（`cmp-maskfull`）用 `mask_loss_until=1.0` 训练，
> 网格仍把碗和一大片桌面包在一起 —— **质量瓶颈在模型侧的背景污染，不在提取算法**。
> 网格预览图（`preview_mesh_iso.png`）可肉眼确认。

> 「轮廓 IoU」是本项目自建的可量化指标（本机 open3d 的离屏渲染在无窗口环境返回全黑，
> "看图评估"不可用）。**单一指标不能作为唯一判据** —— 早期一条路线深度误差更小
> （1.77%）却只覆盖物体 1/5 的面积，IoU 才 0.24。要**同时看覆盖度与精度**。

**测试**：`30 passed`（含真实 COLMAP 二进制解析、mask 阶段契约、Web 路由、GLB 规范校验、导出选择逻辑）。

---

## 二、硬件与关键约束（本机实测）

| 项 | 值 | 影响 |
|---|---|---|
| GPU | RTX 5060 Laptop 8GB，Blackwell **sm_120** | **CUDA 必须 ≥12.8**；文档里的 CUDA 11.8 不可用 |
| torch | 2.11.0+cu128（内核架构含 `sm_120`） | 必须 cu128+ 轮子 |
| CUDA | **便携版 12.8.93（免管理员，358MB）** | 见「便携 CUDA」一节 |
| 内存 | 15.2GB | 大场景避免 exhaustive 匹配与训练并发 |
| COLMAP / FFmpeg | 3.12（CUDA）/ 9.0.2，项目自带 | 免安装 |

---

## 三、快速开始

```bat
:: 1) 环境（虚拟环境 + 核心依赖）
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1

:: 2) GPU 依赖（torch cu128 + gsplat，torch 约 2.6GB）
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Gpu

:: 3) 分割后端（FastSAM / rembg）
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Segment

:: 4) 自带外部工具（免安装、断点续传）
scripts\download.cmd colmap
scripts\download.cmd ffmpeg

:: 5) 环境自检（会打印待办清单）
scripts\gsflow.cmd doctor
```

桌面快捷方式（2 个）：

```bat
env\venv\Scripts\python.exe scripts\make_shortcuts.py
```
- **gsflow Web GUI** —— 双击启动图形界面并打开浏览器
- **gsflow 命令行** —— 打开项目根目录的命令行

---

## 四、便携 CUDA（不需要管理员权限）

官方 CUDA 安装包本质是 7z 自解压包，Windows 自带 `bsdtar`（`tar.exe`）能读，所以可以只解出
编译所需的三个子包，拼成 `tools\cuda`：

```bat
scripts\build_portable_cuda.cmd
```

结果：`tools\cuda`（**358 MB，含 nvcc 12.8.93**），零管理员权限。

> 为什么需要：gsplat 与官方 3DGS 光栅化器都要用 nvcc 现场编译 CUDA 内核；
> 没有 nvcc 时 gsplat 只会打印一句 `No CUDA toolkit found` 然后静默禁用 GPU。
> 完整安装包 3.2GB 且需要管理员（实测还遇到过 `0xE0E00019` 安装器失败），
> 便携版把这两件事都绕开了。

---

## 五、CLI 端怎么训练

训练阶段必须走 **GPU 包装器**，因为它会先加载 MSVC 环境（nvcc 需要 `cl.exe`）再设置便携 CUDA：

```bat
:: 0) 先确保便携 CUDA 存在
scripts\build_portable_cuda.cmd

:: 1) 完整流程（抽帧 → 清理 → 重建 → 训练 → 导出 → 报告）
scripts\gsflow_gpu.cmd run --video D:\data\room.mp4

:: 2) 只训练（已有 COLMAP 结果的作业，续跑）
scripts\gsflow_gpu.cmd run --job-id 20261002-222324-eaf09b --stages train --force train

:: 3) 小步验证（先确认能编出内核，再跑长的）
scripts\gsflow_gpu.cmd run --job-id <job> --stages train --force train ^
    --set train.iterations=300 --set train.resolution=400

:: 4) 8GB 显存上的推荐参数
scripts\gsflow_gpu.cmd run --images D:\data\photos ^
    --set train.iterations=30000 --set train.resolution=1400 --set train.cap_max=1200000

:: 5) 训练 + 导出
scripts\gsflow_gpu.cmd run --job-id <job> --stages train,export --force train

:: 6) 只重跑导出（不需要 GPU）
scripts\gsflow.cmd export <job_id>
```

常用训练参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `train.backend` | `gsplat` | `gsplat`（推荐/免编译）/ `official`（需自编译）/ `external`（自备脚本） |
| `train.iterations` | 30000 | 8GB 卡建议 15000~30000 |
| `train.resolution` | 1600 | 长边；OOM 会自动按 1600→1400→1200→1000→800→600 降档重试 |
| `train.cap_max` | 600000 | 高斯数量上限（显存守护），8GB 建议 ≤120 万 |
| `train.densify_until` | **4000** | 密度控制窗口上界。官方默认 15000；实测改 4000 后高斯数**少 68%** 且不再过训练退化（PSNR 基本持平） |
| `train.mask_loss_until` | **1.0** | 掩码损失生效的迭代占比。**1.0=全程约束在掩码内**，模型轮廓更干净、网格更准 |
| `train.mask_loss_mode` | composite | `composite`（掩码外把 **GT 合成的背景色**，损失在全图算）/ `inside`（只在掩码内取均值，旧行为）。⚠️ **2026-10 修了一个顺序 bug**：`l1` 曾在"替换 GT"之前算好，导致 `composite` **对 L1 完全无效**（只影响 SSIM），背景被当主体拟合。修复后掩码外渲染亮度 0.904 → **0.005**。**连带后果：全图 PSNR/SSIM 不再是画质指标**（背景被有意压黑），训练器改为同时输出 **`psnr_masked` / `ssim_masked`**（只统计掩码内）。项目里 27~31 dB 那类**历史全图 PSNR 与新指标不可直接比较**（`docs/03` G123）|
| `train.sh_degree` | 3 | 球谐阶数，降低可省显存 |
| `train.eval` | true | 是否留出评估集算 PSNR |

排障入口：

```bat
scripts\build_gsplat_kernels.cmd          :: 单独触发内核编译（VERBOSE=1，能看见真实报错）
env\venv\Scripts\python.exe scripts\debug_train.py --iterations 100   :: 打完整异常栈
scripts\gsflow.cmd jobs                    :: 看作业状态与失败原因
```

> **关于 MSVC 环境**：`scripts\gsflow_gpu.cmd` 会先 `call vcvars64.bat` 再启动 Python。
> 现在已经**不需要**依赖这一点 —— `paths.apply_runtime_env()` 会在 Python 内自动
> 装配便携 CUDA 与 MSVC 环境（`vcvars64.bat` 的导出被解析并装入当前进程），
> 所以 CLI / GUI / 测试三条入口的编译环境**完全对等**，
> `scripts\gsflow.cmd` 与 `scripts\web.cmd` 也都能编译内核。

---

## 五之二、三角网格与纹理导出（把"点云"变成"有实感的实体"）

```bat
:: 网格 + 纹理烘焙（默认网格方法是泊松）
scripts\gsflow_gpu.cmd run --job-id <job> --stages export --force export ^
    --set export.formats=mesh --set export.mesh_texture=true

:: 只重跑导出、不重训（复用已有 04_train）
scripts\gsflow_gpu.cmd run --job-id <job> --stages export --force export --set export.formats=mesh
```

导出格式（`export.formats`，逗号分隔）：

| 值 | 产物 | 说明 |
|---|---|---|
| `glb_points` | `model.glb` | 点云式 glTF（`POINTS` 图元 + 顶点色） |
| `spz` | `model.spz` | 需装 `spz` 包；否则跳过 |
| `obj` | `model.ply` | 原始高斯 PLY |
| `viewer_html` | `viewer.html` | 自带 three.js 查看器（单文件） |
| **`mesh`** | `model_mesh.obj/.ply/.glb` | **三角网格**（`TRIANGLES` 图元 + 顶点色），可直接进 Blender / Unity / three.js / Windows「3D 查看器」 |
| `mesh` + `mesh_texture=true` | `model_mesh_textured.glb`、`model_mesh_texture.png`、`model_mesh_uv.obj/.mtl` | **带贴图的网格**（贴图内嵌在 glb 里） |

网格相关参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `export.mesh_method` | `2dgs` | `2dgs`（深度反投影 + 多视角一致性投票，法线最干净）/ `poisson` / `tsdf` |
| `export.mesh_width` | 640 | 渲深度用的分辨率（越大越慢、越细） |
| `export.mesh_max_views` | 80 | 参与融合的视角上限 |
| `export.mesh_poisson_depth` | **8** | 泊松八叉树深度。**更深不等于更好**：同模型只改它，相邻面法线翻转率 dd-0 = 1.7%(8) / 17.8%(9) / **43.5%(10)**，dd-15 = 2.1%(8) / 22.4%(9)，面数与 IoU 几乎不变 |
| `export.mesh_density_quantile` | 0.001 | 密度裁剪分位。**极敏感**：`0.03` 会把轮廓 IoU 从 0.68 砍到 0.32（连通真实表面一起裁了） |
| `export.mesh_crop_quantile` | **1.0** | 网格聚焦的分位数裁剪。**`1.0` = 只用 `mesh_crop_opacity` 过滤、不按分位数裁**。⚠️ 曾默认 0.95，是一条真实事故：3DGS 把容量放在细节密集处，稀疏但真实的屋檐/翼角会被分位数优先裁掉 ⇒ 网格 bbox 只有高斯的 25%/22%/15%、表面积仅 1.72（修好后 18.10），导出的网格只是中央一块扭曲碎片。背景污染交给 `mesh_crop_opacity`（`docs/03` G119）|
| `export.mesh_trim_sigma` | 2.5 | 按"到输入点距离"裁掉泊松桥接面。`1.0` 能把翻转降到 3.2% 但只剩 1.6 万面（裁太狠） |
| `export.mesh_consistency_min_views` | 2 | 多视角一致性投票门槛。`3` 把翻转降到 12.6%，但 **IoU 掉 0.060**（靠删面换来的） |
| `export.mesh_presimplify_ratio` | 10.0 | 预简化比例（先降规模再平滑） |
| `export.mesh_target_triangles` | 400000 | QEM 简化目标面数（0=不简化） |
| `export.mesh_smooth_iters` | 5 | Taubin 平滑迭代（比拉普拉斯好：不收缩体积） |
| `export.mesh_texture` | false | 是否烘焙贴图（>8 万面时 xatlas 超线性变慢） |
| `export.mesh_texture_size` | 2048 | 贴图边长 |
| `export.formats` | glb_points, mesh, viewer_html, **splat** | 导出格式。`splat` 为高斯泼溅文件（32 字节/高斯，11 维：位置+尺度+旋转+不透明度）。**查看器还会直接读训练的 3DGS PLY（59 维，含 SH 48）做视角相关着色**，与 `preview_*.png` 一致；glTF 点云只有 6 维、必然稀疏扁平（见 `docs/03` G108）|
| `export.mesh_texture_reject_black` | true | 拒绝采到**纯黑背景**的像素（`mask.mode=black` 的训练图背景是 0）。实测 `final-v1` 黑色 texel **35.99% → 2.30%**。阈值是 `6/255`（注意图像在烘焙里是 **0~1 尺度**，写成 `> 6` 会永远为假 —— 我踩过，见 `docs/03` G104）|
| `export.mask_dir` | （自动） | 纹理烘焙用的**前景掩码**目录。留空=自动查找：本作业 `02_masks` → 同级作业（**必须通过内容校验**）。同级里有 40+ 个作业文件名一致但其实是不同数据集，只按文件名会选错，**错掩码比没掩码更糟**（见 `docs/03` G107）|
| `train.rasterize_mode` | classic | 光栅化模式：`classic` / `antialiased`（Mip-Splatting 低通滤波）。**实测 `antialiased` 在训练/评估同分辨率时明显更差**（28.79 → 27.72，−1.07 dB），故默认 classic |
| `train.absgrad` | false | AbsGS 绝对梯度致密化。4 次重复实测 +0.098 dB，95% CI [−0.114, +0.310] **不显著**，故默认关闭（见 `docs/03` G106）|
| `export.mesh_drop_unseen` | true | **删除"从未被任何相机看到"的网格面**（泊松重建的内部垃圾几何）。实测 25.6% 的面从未被看到且散布在物体内部，它们白占 UV 空间却烘不出颜色 ⇒ **纹理黑洞**。判据是"是否被相机看到过"，**不是**"落在前景掩码内的比例"（后者在转台素材上原理错误，见 `docs/03` G101）|
| `export.mesh_min_views` | 1 | 保留"被至少这么多视角看到"的面 |
| `train.lambda_lpips` | 0 | 感知损失权重（自己实现的 VGG16 LPIPS，见 `docs/05` §5.3）。**默认关闭**：3 次重复实测三指标均不显著，且方差约 3 倍 |
| `train.lambda_depth_distortion` | 0 | 深度失真正则（2DGS 的 λd）。**权重必须按本项目损失口径实测校准**，不能照抄论文的 1000。**注意**：单次测得 +0.199 dB，但 3 次重复后 **+0.026 dB、不显著**（见 `docs/03` G90-B）|
| `train.opacity_cleanup` | none | **不透明度治理方案**（GUI 下拉框与 CLI `--opacity-cleanup` 共用同一份映射，见 `trainers/opacity_cleanup.py`）：`none`（默认最稳）/ `decay`（C·平滑衰减）/ `recovery`（A·恢复感知剪枝）/ `utilization`（B·利用率剪枝）/ `reset`（官方周期重置，实测最差）。实测见 `docs/03` G84 |
| `train.densify_criterion` | grad2d | 致密化判据来源：`grad2d`（官方口径：屏幕空间梯度 / 可见次数）/ `grad3d`（旧：世界空间梯度）。**两者量纲差 45 倍、选中的高斯 IoU 仅 0.065**（`docs/03` G90-C）。**实测受控对比（同数据、2 重复、95% CI，`docs/03` G117）**：三者**质量无显著差异**，但同等画质下 `grad3d` 要多 **48%** 高斯、阈值降到 1e-4 要多 **89%** 高斯 ⇒ **`grad2d` + 官方默认 `2e-4` 是最省的配置**。复核用 `scripts/ab_densify_rasterize.py`，并用 `scripts/check_train_used_config.py` 确认判据真的生效（G109 曾让 grad2d 静默退回 grad3d）|
| `capture.mode` | auto | **拍摄模式**（GUI 下拉框与 CLI `--capture-mode` 共用同一份映射，见 `gsdata/capture_mode_params.py`）：`auto`（按背景帧间差异自动判定）/ `turntable`（物体转、相机固定 ⇒ **保留背景** `mask.mode=keep`）/ `orbit`（相机绕行 ⇒ 抠背景）。实测两类素材的比值相差 **34 倍**（`docs/03` G91）|
| `mask.mode` | black | `black`（背景填黑）/ `inpaint`（纹理补全）/ **`keep`（保留原始背景 —— 转台拍摄必须用）** |
| `train.densify_grad2d_threshold` | 2e-4 | `densify_criterion=grad2d` 时的阈值，**官方默认值**（`gsplat/strategy/default.py` 的 `grow_grad2d`）。只有 grad2d 口径下这个数值才有意义 —— grad3d 的梯度量纲不同，套用会把高斯数推高约 48%（`docs/03` G90、G117）。**实测把它减半到 1e-4 没有质量收益**，只是让高斯数 +89%（G117）⇒ 不要随手调低 |
| `frames.auto_crop` | true | **自动裁剪（必须在 COLMAP 之前做）**。判据是**帧间时间方差**：转台时物体在动、背景静止；相机绕行时全图都变 ⇒ 包围盒≈全图 ⇒ 自动不裁。不需要掩码或分割模型。实测 IMG3 物体只占 8.5% 像素，裁剪后前景 81% ⇒ PSNR 20.74 → **28.58 dB**（`docs/03` G112）|
| `frames.crop` | （空） | 手动裁剪框 `x,y,w,h`（原图坐标）。填了就**优先于**自动裁剪 |
| `train.auto_crop` | true | 兼容旧行为用的开关：是否允许 train 阶段在掩码同步后**再**求一次并集包围盒并裁剪（已裁过则跳过）|
| `train.auto_crop_in_train` | false | **默认关闭**：在 train 阶段裁剪太晚 —— COLMAP 已经用全图建完整个场景（会把整个房间都重建出来，点云 bbox 13.7×1.6×8.5、离质心最远 10.63，而正确做法只有 5.7×1.0×5.4 / 4.51），初始化高斯散布全场景后 PSNR 只有 19.97 dB。裁图请在 `frames` 阶段（`frames.auto_crop`）做（`docs/03` G112）|

> 网格流程（对应社区标准做法）：渲深度 → 按不透明度/深度一致性过滤 →
> **反投影成紧贴表面的稠密点云** → 体素下采样均匀化 → 泊松重建 →
> Taubin 平滑 → QEM 简化 →（可选）UV 展开 + 纹理烘焙。
>
> **关键点**：不要把高斯中心直接当点云喂给泊松。高斯中心是"分布在表面附近的
> 三维体"，不是薄壳，泊松会鼓包 + 过拟合 —— 实测那条路线网格只有物体的 1/5 大
> （轮廓 IoU 0.24），而深度反投影路线是 **0.66**。
>
> **密度裁剪（`density_quantile`）要多加小心**：它的本意是"用密度图把低密度区域的
> 包络面删掉"，但全局分位数会**误伤观测较少的侧面**（正对相机的区域密度高、
> 侧面密度低，一个全局阈值切不平）。实测 `0.03` 直接把 IoU 从 0.68 砍到 0.32。
> 泊松在"完全无数据"处的密度接近 0，所以阈值要取**很小**的值；默认 `0.0` 表示不裁。

排障与调参脚本：

```bat
env\venv\Scripts\python.exe scripts\tune_mesh.py --job <job>              :: 在已有模型上秒级扫网格参数 + IoU 打分
env\venv\Scripts\python.exe scripts\verify_mesh_quality.py --job <job>    :: 轮廓 IoU + 深度一致性
env\venv\Scripts\python.exe scripts\dump_glb.py --path <file.glb>         :: 打印 GLB 结构，查"声明与内容是否自洽"
env\venv\Scripts\python.exe scripts\check_mesh_indices.py --mesh <f.obj>  :: 查索引越界 / UV 数量
```

---

## 六、Web GUI

```bat
scripts\web.cmd                :: 默认 http://127.0.0.1:8787，会自动开浏览器
scripts\web.cmd --port 9000    :: 换端口
```

界面 5 个页签：

1. **新建作业** —— 输入路径、勾阶段、调关键参数；含**「3.5 网格导出」卡片**
   （网格方法 / 导出格式 / 泊松深度 / 目标面数 / 平滑迭代 / 深度分辨率 / 视角上限 /
   密度裁剪分位 / 纹理烘焙开关 / 贴图分辨率）；右下角实时显示**等价的 CLI 命令**
2. **作业与进度** —— 实时日志 + 进度条 + 取消；历史作业可回看日志与产物链接
3. **环境自检** —— 等于 `gsflow doctor`，上方是待办清单
4. **设置 / API Key** —— HuggingFace Token、SAM3 API Key、其它预留 Key
   （如 SD-inpaint）、HF 镜像、代理，写入 `config/credentials.json`
   （已 gitignore，界面只回显尾 4 位）
5. **说明** —— 流程与显存建议

> GUI 与 CLI 共用同一个 Orchestrator、同一份 `job.json`、同一套事件总线：
> GUI 提交的作业在 `gsflow jobs` 里能看到，反之亦然。
>
> 这条红线有**静态检查守门**：`scripts\check_gui_wiring.py` 校验 HTML 控件与
> `app.js` 引用一一对应，防止"界面填了但提交时没带上"这类静默失真。

---

## 七、目录结构

完整文件树见 [`docs/04-文件树.md`](docs/04-文件树.md)（由 `scripts/make_tree.py` **自动生成**，勿手工编辑）。

```
3dgs-and-ai/
├─ config/      config.toml（可提交）、credentials.json（本地密钥）
├─ data/        你的原始素材（视频/图片）
├─ docs/        00 环境核查与规划 / 01 架构 / 02 操作手册 / 03 踩坑记录
│               04 文件树（自动生成） / 05 训练框架
├─ scripts/     环境搭建、工具下载、启动器、验证/诊断/实验脚本
├─ src/gsflow/  流程核心：pipeline / stages / segment / trainers / gsdata / tools / web
├─ tests/       pytest 回归
├─ tools/       自带 colmap / ffmpeg / 便携 cuda
├─ env/         虚拟环境、缓存、临时目录
└─ workspace/   每个作业一个子目录（01_images_raw … 06_reports）
```

各模块行数与职责见 [`docs/04-文件树.md`](docs/04-文件树.md)；
训练细节（初始化/损失/密度控制/顺序红线/评估协议）见
[`docs/05-训练框架.md`](docs/05-训练框架.md)。

---

## 八、测试与验证脚本

```bat
:: 全部测试
env\venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider

:: 以下脚本都不需要真实数据集 / 分割权重 / GPU
env\venv\Scripts\python.exe scripts\status.py                      :: 环境速览
env\venv\Scripts\python.exe scripts\synthetic_e2e.py --views 36 --stages frames,mask,colmap
env\venv\Scripts\python.exe scripts\verify_mask_stage.py           :: mask 阶段 + IoU
env\venv\Scripts\python.exe scripts\mask_selftest.py               :: 各分割后端对比
env\venv\Scripts\python.exe scripts\verify_export_stage.py         :: 导出 + GLB 规范校验
env\venv\Scripts\python.exe scripts\verify_portable_cuda.py        :: 便携 CUDA 编译内核
env\venv\Scripts\python.exe scripts\check_gui_wiring.py            :: GUI 控件与 app.js 接线一致性
env\venv\Scripts\python.exe scripts\gui_job_smoke.py --port 8787   :: 经 GUI API 跑一次作业
```

需要 GPU / 已训练模型的脚本：

```bat
scripts\gpu_python.cmd scripts\verify_mesh_quality.py --job <job>   :: 轮廓 IoU + 深度一致性
scripts\gpu_python.cmd scripts\tune_mesh.py --job <job>             :: 网格参数扫描 + IoU 打分
scripts\gpu_python.cmd scripts\gui_mesh_smoke.py --port 8787 --job <job>  :: 经 GUI 出网格+贴图
```

排障脚本（排查环境与二进制格式问题）：

```bat
env\venv\Scripts\python.exe scripts\diag_msvc_env.py        :: MSVC 环境解析
scripts\gpu_python.cmd scripts\diag_gsplat_preload.py       :: gsplat 内核预加载（应在 1s 内）
env\venv\Scripts\python.exe scripts\dump_glb.py --path a.glb        :: GLB 结构
env\venv\Scripts\python.exe scripts\check_mesh_indices.py --mesh a.obj :: 网格索引
```

---

## 九、文档

| 文档 | 内容 |
|---|---|
| **[交接与下一步](docs/06-交接与下一步.md)** | **新会话先读这一页**：三十秒现状、明天要做的事（含命令与验收判据）、出问题第一站、判据红线 |
| [环境核查与实施规划](docs/00-环境核查与实施规划.md) | 开工前的实测与判断（保留留痕）；**§六 是"计划 vs 实际"对照表** |
| [架构说明](docs/01-架构说明.md) | **含 4 张 Mermaid 图**：分层总览、数据流（到**五个**查看形态）、架构红线、模块地图、扩展点、**11 条实验纪律** |
| [操作手册](docs/02-操作手册.md) | 一次性准备、CLI/GUI 用法、**模型页五模式**、参数速查、**质量数字怎么看**、典型工作流 |
| [踩坑记录](docs/03-踩坑记录.md) | 全过程踩坑与解法（编号到 **G124**）；每条都带实测数字，负面结论也记录 |
| [文件树](docs/04-文件树.md) | 工程结构与作业产物布局（**自动生成**，含各模块行数；`static/` 与 `vendor/` 已展开） |
| [训练框架](docs/05-训练框架.md) | **训练代码提炼**：数据准备、参数初始化、损失口径（含掩码顺序修正）、密度控制、顺序红线、评估协议（含掩码内指标）、改代码检查清单 |
| [AGENTS.md](AGENTS.md) | 给协作者的既定事实与架构红线 |

> **文档一致性由脚本保证**（改完代码/默认值必须跑）：
> `python scripts\check_docs_consistency.py` 一次做 5 项检查 ——
> 代码默认值 ↔ `TrainConfig` ↔ `config.toml` ↔ README 参数表 ↔ mermaid 结构 ↔
> **关键概念覆盖**（`check_docs_concepts.py`，自带负向自测）。
> 加最后一项是因为只校验数值时出现过"默认值全一致 ✅、但文档 0 处提到
> `psnr_masked`"的漂移（`docs/03` G123）。

**踩坑记录里值得单独一提的几条**（都是花了很久才定位的）：

| 编号 | 内容 | 一句话教训 |
|---|---|---|
| G17 | 泊松的 `density_quantile=0.03` 把网格质量砍半（IoU 0.68→0.32） | **先建立便宜的量化评分，再调参**；否则只能"改一行跑一次看图猜" |
| G18 | 网格必须走**深度反投影**，不能拿高斯中心直接喂泊松 | 高斯中心是三维体不是薄壳；实测 IoU 0.24 → 0.65 |
| G19 | 网格 GLB 的三个坑：uint32 索引 / y-z 取反 / 逆时针绕序 | 索引越界会让查看器**黑屏且不报错** |
| G22 | GLB 的位置数据被写了两遍（`parts=[pos]` 又 `_add(pos)`） | **断言全过但结果不对 ⇒ 断言的层次选错了**，要断言最终产物 |
| G23 | open3d 写出的"带贴图 glb"其实没有贴图 | 第三方库的静默降级要自己验证，别信日志 |
| G25 | GUI 与 CLI 的编译环境不对等（GUI 用不了 gsplat） | **"能不能用"不能取决于用户从哪个脚本启动** |
| G26 | gsplat 每次 import 都重编内核（80~140s → 0.2s） | 四个坑里有三个是**顺序**问题；靠"打印每步耗时"定位 |
| **G34** | **检查点保存被不透明度重置污染，导出的 PLY 渲染全黑** | **"训练器说好、下游说坏" ⇒ 查两者之间的落盘/读盘环节** |

---

## 十、设计红线

1. **流程核心不依赖 UI**：`src/gsflow/pipeline` + `stages` 是唯一事实来源；CLI 与 Web 只是入口。
2. **阶段之间只通过磁盘产物 + `job.json` 通信**：随时跳过、续跑、单阶段重跑。
3. **外部工具与模型可插拔**：分割后端（classical/fastsam/rembg/sam2/sam3）、
   训练后端（gsplat/official/external）都是接口。
4. **失败要能自救**：缺工具 / 缺凭据 / COLMAP 失败都给出可执行的修复建议，而不是堆栈。
5. **两端同构**：GUI 的每次提交都能翻译成一条可直接执行的 CLI 命令（有测试守门）。
6. **入口环境对等**：CLI / GUI / 测试三条入口自动获得同样的编译环境
   （`paths.apply_runtime_env()` 内装配便携 CUDA + MSVC），
   不允许出现"某个入口少了一层包装所以某些功能不可用"。
7. **产物要能被通用工具打开**：GLB 必须过 glTF 2.0 结构校验（含索引越界检查），
   网格要能直接拖进 Blender / Unity / three.js / Windows「3D 查看器」。
