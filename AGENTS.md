# AGENTS.md —— 本工作区给 AI 协作者的说明

## 用户定的规则（原样保留）
1. 需要用 python 或安装相应的 python 库时，使用 python 虚拟环境；如果没有现成的虚拟环境，则在此工作文件夹下创建一个，做好与系统 python 的隔离。
2.  因权限失败时可以优先向我提权。

## 本机的既定事实（实测，别再重新试错）

### 环境
- 工作区：`F:\3dgs-and-ai`（DSH 沙箱唯一授权可写的根）。
- `D:\3dgs-workspace` 是指向本工作区子目录的 **目录链接（junction）**：
  `env / workspace / tools / models / downloads` 五个名字。真实目录都在 F 盘。
- 虚拟环境：`F:\3dgs-and-ai\env\venv`（= `D:\3dgs-workspace\env\venv`），Python 3.13.11，
  由 **uv** 创建，源码通过 `site-packages\gsflow-dev.pth` 挂载（不是 pip install -e）。
- GPU：RTX 5060 Laptop 8GB，Blackwell **sm_120**，驱动 573.22 ⇒ **CUDA 必须 ≥12.8，
  禁止 CUDA 11.x**；torch 必须用 cu128+ 轮子（≥2.7 才带 sm_120 内核）。

### 沙箱踩坑（会导致"看不懂的"失败，务必按下面做）
1. **TEMP/TMP 必须指到工作区内**：`F:\3dgs-and-ai\env\tmp`。
   否则 `ensurepip`、`pip`、`uv`、`pytest` 的临时目录清理会 `PermissionError: WinError 5`。
2. **pip 的 PEP517 构建（`pip install -e .`）在本机会失败**（同样 WinError 5）。
   要装源码就用 `.pth` 挂 `src`，或统一用 `uv pip install --python env\venv\Scripts\python.exe`。
3. **系统 `python -m venv` / `ensurepip` 不可用**（chmod 沙箱临时目录被拒），用 uv 建环境。
4. **`pwsh.exe` 不存在**（只有 `powershell.exe`），且脚本执行策略是 `RemoteSigned`：
   跑 `.ps1` 用 `powershell -ExecutionPolicy Bypass -File scripts\xxx.ps1`。
5. **curl.exe / PowerShell 的 HTTPS 请求会失败**（Schannel `SEC_E_NO_CREDENTIALS`）。
   下载一律用 Python：`scripts\download_tools.py` / `download_weights.py` 或 `urllib.request`。
5b. **`.ps1` 必须写成纯 ASCII + UTF-8 BOM + CRLF**。Windows PowerShell 5.1 对无 BOM 的
   `.ps1` 按系统 ANSI(GBK) 解码，含中文的注释会吞掉字符串结束符并报一堆假语法错误。
   校验：前 3 字节 `EF BB BF`；`[Parser]::ParseFile($f,[ref]$null,[ref]$errors)` 无错。
5c. **不要用 PowerShell 的 `Get-Content`/`Set-Content` 改含中文的 UTF-8 源码**（会按 GBK 写坏）。
   要改文件用编辑工具。
5d. **第三方库的目录也要关进工作区**（导入前设置）：
   `YOLO_CONFIG_DIR`（ultralytics，默认 AppData\Roaming）、`MPLCONFIGDIR`、
   `REMBG_HOME`/`U2NET_HOME`（rembg，默认 %USERPROFILE%\.rembg）。
   已统一在 `paths.runtime_env()` 里，CLI/GUI 入口会自动应用。
6. **不要在 `env\pytest-tmp` 之外用 pytest 的 `tmp_path`**：其 session 收尾会 WinError 5。
   已改为自定义 `workdir` fixture（放 `env\pytest-tmp`）。
7. 控制台中文乱码：给子进程设 `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`，`.cmd` 里先 `chcp 65001`。
8. `huggingface.co` 直连不稳定（有时通、有时超时）⇒ 配置里留了 `hf_endpoint` 镜像与代理开关。

### 常用命令（都用工作区虚拟环境）
```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Gpu    # 建环境 + torch(cu128) + gsplat
scripts\gsflow.cmd doctor                                              # 环境自检
scripts\gsflow.cmd run --images D:\data\photos --stages frames,mask
scripts\web.cmd                                                        # 启动 Web GUI (127.0.0.1:8787)
env\venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

### 架构红线（改动时别破坏）
- 流程核心在 `src\gsflow\pipeline` + `stages`，**不依赖 UI**；CLI 与 Web 只是两个入口。
- 阶段之间只通过磁盘产物 + `job.json` 通信，支持跳过/续跑/单阶段重跑。
- 新增外部工具要走 `tools\locate.py` 的适配器；新增分割后端实现 `segment\base.py:Segmenter`；
  新增训练后端实现 `trainers\base.py` 的协议。
  