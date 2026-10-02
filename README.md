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
| ④ 3DGS 训练（gsplat） | ✅ 已实测 | 免管理员便携 CUDA 编译内核通过；30k 迭代跑通并产出 PLY |
| ⑤ 导出（GLB / 查看器 / 预览图） | ✅ 已实测 | 纯 Python；GLB 通过 glTF 2.0 结构校验 |
| ⑥ 质量报告 | ✅ 已实测 | 注册率、掩码占比、清晰度、PSNR、改进建议 |

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
| `train.cap_max` | 1500000 | 高斯数量上限（显存守护），8GB 建议 ≤120 万 |
| `train.sh_degree` | 3 | 球谐阶数，降低可省显存 |
| `train.eval` | true | 是否留出评估集算 PSNR |

排障入口：

```bat
scripts\build_gsplat_kernels.cmd          :: 单独触发内核编译（VERBOSE=1，能看见真实报错）
env\venv\Scripts\python.exe scripts\debug_train.py --iterations 100   :: 打完整异常栈
scripts\gsflow.cmd jobs                    :: 看作业状态与失败原因
```

> **踩坑提醒**：不要用 `scripts\gsflow.cmd` 跑训练。它不带 MSVC 环境，
> nvcc 会报 `Cannot find compiler 'cl.exe' in PATH`。训练一律用 `scripts\gsflow_gpu.cmd`。

---

## 六、Web GUI

```bat
scripts\web.cmd                :: 默认 http://127.0.0.1:8787，会自动开浏览器
scripts\web.cmd --port 9000    :: 换端口
```

界面 5 个页签：

1. **新建作业** —— 输入路径、勾阶段、调关键参数；右下角实时显示**等价的 CLI 命令**
2. **作业与进度** —— 实时日志 + 进度条 + 取消；历史作业可回看日志与产物链接
3. **环境自检** —— 等于 `gsflow doctor`，上方是待办清单
4. **设置 / API Key** —— HuggingFace Token、HF 镜像、代理、SAM3 Key，写入
   `config/credentials.json`（已 gitignore，界面只回显尾 4 位）
5. **说明** —— 流程与显存建议

> GUI 与 CLI 共用同一个 Orchestrator、同一份 `job.json`、同一套事件总线：
> GUI 提交的作业在 `gsflow jobs` 里能看到，反之亦然。

---

## 七、目录结构

完整文件树见 [`docs/04-文件树.md`](docs/04-文件树.md)（由 `scripts/make_tree.py` 生成）。

```
3dgs-and-ai/
├─ config/      config.toml（可提交）、credentials.json（本地密钥）
├─ data/        你的原始素材（视频/图片）
├─ docs/        00 环境核查与规划 / 01 架构 / 02 操作手册 / 03 踩坑记录 / 04 文件树
├─ scripts/     环境搭建、工具下载、启动器、验证脚本
├─ src/gsflow/  流程核心：pipeline / stages / segment / trainers / gsdata / tools / web
├─ tests/       29 项测试
├─ tools/       自带 colmap / ffmpeg / 便携 cuda
├─ env/         虚拟环境、缓存、临时目录
└─ workspace/   每个作业一个子目录（01_images_raw … 06_reports）
```

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
env\venv\Scripts\python.exe scripts\gui_job_smoke.py --port 8787   :: 经 GUI API 跑一次作业
```

---

## 九、文档

| 文档 | 内容 |
|---|---|
| [环境核查与实施规划](docs/00-环境核查与实施规划.md) | 实测软硬件、与原始文档的偏差、分阶段计划 |
| [架构说明](docs/01-架构说明.md) | 分层、模块地图、阶段契约、扩展点 |
| [操作手册](docs/02-操作手册.md) | 一次性准备、CLI/GUI 用法、参数速查、典型工作流 |
| [踩坑记录](docs/03-踩坑记录.md) | 全过程踩坑与解法（沙箱、CUDA、COLMAP、编码等） |
| [文件树](docs/04-文件树.md) | 工程结构与作业产物布局 |
| [AGENTS.md](AGENTS.md) | 给协作者的既定事实与架构红线 |

---

## 十、设计红线

1. **流程核心不依赖 UI**：`src/gsflow/pipeline` + `stages` 是唯一事实来源；CLI 与 Web 只是入口。
2. **阶段之间只通过磁盘产物 + `job.json` 通信**：随时跳过、续跑、单阶段重跑。
3. **外部工具与模型可插拔**：分割后端（classical/fastsam/rembg/sam2/sam3）、
   训练后端（gsplat/official/external）都是接口。
4. **失败要能自救**：缺工具 / 缺凭据 / COLMAP 失败都给出可执行的修复建议，而不是堆栈。
5. **两端同构**：GUI 的每次提交都能翻译成一条可直接执行的 CLI 命令（有测试守门）。
