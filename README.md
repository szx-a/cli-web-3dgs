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
2. **双掩码损失**：`mask_loss_mode=composite` + `bg_alpha_reg`，把"物体外不许有高斯"变成可优化项；
   同时保留原始掩码用于背景抑制（不侵蚀前景）。
3. **退火式"推实"正则**（`opacity_crisp_reg=0.02` + `warmup=0.7`）：把"淡高斯云"推成实心表面，
   **`op>0.9` 15%→22%，SSIM 只 −0.0003**；全程施加会掉 0.0043 并出亮斑，所以必须退火。
4. **深度语义统一**：训练侧法线/失真正则从 `Σwz`（未归一化）改为 **ED（期望深度）**，与导出反投影一致。
   旧实现两者相差 1/alpha 倍 —— 各段都能跑，但**语义不自洽**（已加测试锁定）。
5. **网格水密化 + ED 精修**：边界边少 ≠ 封闭；`fill_holes` 按边界环补面（逆原面绕序），
   再用 ED 深度做**回溯式**顶点精修（ICP 风格阻尼，残差单调下降）。
6. **负结果也写进文档**：LPIPS 每 16 步才算是**错的**（梯度与 L1/SSIM 打架，实测 LPIPS 反而变差），
   必须每步算；`opacity_init=0.5` 无效（淡高斯是目标函数的强吸引子）。

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

```bat
:: 视频
scripts\gsflow.cmd run --video data\IMG5.mp4 --job-id img5 --stages frames,mask,colmap,train,export

:: 照片
scripts\gsflow.cmd run --images D:\photos\object --stages frames,mask,colmap,train,export

:: 质量优先（官方密度控制参数已校准，见"预设"）
scripts\gsflow.cmd run --video data\IMG5.mp4 --preset quality

:: 只重跑训练+导出（复用已有帧/掩码/位姿）
scripts\gsflow.cmd run --job-id img5 --stages train,export --force train,export

:: Web GUI
scripts\web.cmd            :: http://127.0.0.1:8787
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


---

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
