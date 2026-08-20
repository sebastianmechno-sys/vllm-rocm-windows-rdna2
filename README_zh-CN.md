# vllm-rocm-windows-RDNA2-oneclick（中文版）

[![Windows](https://img.shields.io/badge/OS-Windows%2010%2F11-0078D6?style=for-the-badge&logo=windows)]()
[![ROCm](https://img.shields.io/badge/ROCm-7.15%20TheRock-FF0000?style=for-the-badge&logo=amd)]()
[![RDNA2](https://img.shields.io/badge/GPU-RDNA2%20RX%206400%E2%80%936950-000000?style=for-the-badge)]()
[![vLLM](https://img.shields.io/badge/vLLM-0.19.1-00C853?style=for-the-badge)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2Brocm7.15-EE4C2C?style=for-the-badge&logo=pytorch)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](LICENSE)

> English version: [README.md](README.md)

在 **Windows** 上为整个 AMD Radeon **RDNA2 家族**（RX 6000 系列）提供原生的
vLLM + ROCm 7.15（TheRock）运行时 —— **无需 WSL2、无需 NVIDIA、无需编译器**。
一键安装，全部预编译，自带 OpenAI 兼容的聊天服务端，体验对标 NVIDIA 全家桶。
代码与构建流水线同样支持 **RDNA3（RX 7000）** 与 **RDNA4（RX 9000）** ——
这两个家族目前为实验性状态，待硬件验证（见 [docs/MULTIARCH.md](docs/MULTIARCH.md)）。

**已在 AMD Radeon RX 6750 XT 12 GB（gfx1031）实测通过 —— Windows 11 原生 —— 2026 年 8 月**
**RX 6800（gfx1030）：测试进度见 [issue #2](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2/issues/2) —— 待确认**

| 结果 | 数值 |
|---|---|
| rocBLAS FP16 GEMM（原生基准测试） | **25 674 Gflops ≈ 26 TFLOPS** |
| vLLM 解码，Qwen3.5-4B 4-bit | **约 58-62 tok/s**（8.3 → 62.5 = 优化 7.5 倍） |

### 显卡支持矩阵

| 家族 | 显卡 | gfx | Override | 状态 |
|---|---|---|---|---|
| RDNA2 | RX 6400–6950（台式机 + M 版本），Radeon Pro V620 | 1030/1031/1032 | `10.3.1` | ✅ 已在 RX 6750 XT 验证；其余型号待验证（[提交你的验证报告](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2/issues/new?template=gpu_verification.yml)） |
| RDNA3 | RX 7600–7900（台式机 + M 版本） | 1100/1101/1102 | `11.0.0` | ⚠️ 实验性 —— 流水线已就绪，归档尚未发布 |
| RDNA4 | RX 9000 | 1200/1201 | `12.0.0` | ⚠️ 实验性 —— 可行性未确认 |

`INSTALL.bat` 自动检测显卡家族；`INSTALL.bat -Variant rdna3` 可强制指定。

---

## 验证 —— RX 6750 XT 实测输出

### 1. ROCm 识别显卡（原生 Windows 进程，无 WSL）

![gpu detection](assets/01_gpu_detection.png)

### 2. 显卡原始算力 —— 原生 rocblas-bench.exe，FP16 GEMM 4096³

![rocblas bench](assets/02_rocblas_bench.png)

### 3. vLLM 优化后的解码性能

![vllm tps](assets/03_vllm_tps.png)

---

## 系统要求

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10/11（推荐 Windows 11；`tar` 需支持 zstd —— Win11 自动支持） |
| 显卡 | **AMD RDNA2** —— RX 6400 / 6500 / 6600 / 6650 / 6700 / 6750 / 6800 / 6900 / 6950（全部 XT/M 型号），**Radeon Pro V620**（gfx1030）。RDNA3/RDNA4：实验性（见上方支持矩阵）。4B 模型需要 8 GB 以上显存；4 GB 显卡（RX 6400/6500）会自动选择更小的模型 |
| 驱动 | AMD Software: Adrenalin Edition（普通游戏驱动即可） |
| 磁盘 | `C:` 盘需约 25 GB 可用空间 |
| 网络 | 仅在安装时需要（约 6 GB：运行时约 2.3 GB + 模型约 3.8 GB） |
| 权限 | 一次 UAC 点击（安装程序自动提权） |

无需编译器、无需 ROCm 安装包、无需手动配置 —— 一切均为预编译。

## 快速开始（一键安装）

1. 下载本仓库（ZIP 或 `git clone`）。**无需**手动下载发布归档
   （`*.tar.zst`）—— 安装程序会自动从 [Releases](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2/releases)
   页面获取。
2. **双击 `INSTALL.bat`** —— 默认从本仓库的 releases 下载；可传入 GitHub
   用户名使用你自己的 fork，`-Variant` 可强制指定显卡家族。它会检查显卡与磁盘，
   然后安装所有缺失的组件（重复运行始终安全且快速）：

   | 步骤 | 操作 |
   |---|---|
   | 1/6 | 显卡家族检测（rdna2/rdna3/rdna4，未知显卡给出警告）+ Python 3.11.9 |
   | 2/6 | 从 GitHub Releases 下载 4 个归档 → `C:\Python311`、`C:\TheRock`、`C:\vw_*_build` |
   | 3/6 | venv 修复 + torch 自检 |
   | 4/6 | Qwen3.5-4B 4-bit 模型（若已在 HuggingFace 缓存中则跳过） |
   | 5/6 | 写入 `config.bat`（以后修改模型就编辑这个文件） |
   | 6/6 | 验证基准测试 |

3. **`CHAT.bat`** → 在浏览器中打开网页聊天（首次使用会自动启动模型服务端）。
   默认模型**直接回答**（`config.bat` 中 `THINKING=0`）；设置 `THINKING=1` 可让
   模型展示推理过程 —— 使用 Qwen3.5 时推理文本会出现在回答内部（该模板不会把
   推理拆分为独立字段）。Token 实时流式输出并带有 tok/s 计数，全部在你的 AMD
   显卡上本地运行。
4. **`SERVE.bat`** → 单独启动模型服务端（OpenAI 兼容 API，
   地址 `http://127.0.0.1:8000/v1`，相当于 NVIDIA 上的 `vllm serve`）。
   可与任意 OpenAI 客户端配合使用，或直接运行 `CHAT.bat`。
5. **`VERIFY.bat`** → 一次运行全部 3 项验证：ROCm 显卡检测、原生 rocBLAS FP16
   算力（**约 26 TFLOPS**）以及完整 512-token vLLM 基准测试（**约 58-62 tok/s**）。
6. **`UNINSTALL.bat`** → 不再需要时彻底卸载整套环境（保留仓库文件夹本身）。

## 基准测试

| 测试 | 配置 | 结果 |
|---|---|---|
| rocBLAS FP16 GEMM | 4096×4096×4096，rocblas-bench.exe | **25 674 Gflops（≈26 TFLOPS）** |
| vLLM Qwen3.5-4B 解码 | 512 tok，贪心，CUDA graphs | **59.4 tok/s**（最高 62.5） |
| 优化历程 | eager fp16 基线 | 8.3 → 62.5 tok/s（**7.5 倍**） |

### 原生 ROCm vs Vulkan（llama.cpp）

在 Windows 上于 RX 6000 运行 LLM 的另一条路只有 llama.cpp 的 Vulkan 后端
（AMD 从未为这些显卡在 Windows 上提供过 ROCm）。社区实测数据一致表明，在同一
RDNA2 芯片上 Vulkan 只有原生 HIP 吞吐的 **70-80%**，而且 Vulkan 根本无法运行
vLLM（OpenAI 兼容服务、连续批处理、PagedAttention）—— 这正是本方案带来的能力：

| 后端 | vLLM 服务 | Qwen3.5-4B 解码（RX 6750 XT） | OpenAI API / 批处理 |
|---|---|---|---:|
| **本方案（原生 HIP + ROCm 7.15）** | ✅ vLLM 0.19.1 | **59-62 tok/s** | ✅ |
| llama.cpp Vulkan（社区备选） | ❌（仅 llama-server） | 约 40-45 tok/s* | ⚠️ llama.cpp API |

*_按社区普遍报告的 70-80% HIP 吞吐估算；llama.cpp HIP vs Vulkan 的实测对比在
RDNA3（70-80%）与 RDNA2（同一区间）上由 [craftrigs.com](https://craftrigs.com/guides/amd-rocm-llm-inference-support-2026)
完成。_

![progression](results/benchmark.png)

完整优化历程：

| # | 配置 | tok/s |
|---|---:|---:|
| 1 | fp16 eager（基线） | 8.3 |
| 2 | + CUDA graphs + 瘦身 GEMV | 24.4 |
| 3 | + AWQ 4-bit 量化 | 29.9 |
| 4 | + 原生 HIP W4 GEMV 内核 | 35.9 |
| 5 | + M=1 GEMV（用于 lm_head） | 58.1 |
| 6 | + 直写内核路径 | 59.1 |
| 7 | + 权重复制缓存 | **62.5** |

## 工作原理

1. **TheRock** 将 ROCm（HIP 运行时、rocBLAS、Tensile）构建为原生 Windows
   二进制 —— 这正是 ROCm 能在 Windows 上存在的根本原因。
2. `HSA_OVERRIDE_GFX_VERSION=10.3.1` 将任意 RDNA2 显卡（gfx1030/1031/1032）
   伪装为 gfx1031 —— 打包的 rocBLAS/Tensile 库与 `torch_gfx1031.kpack` 即为此
   架构构建；HIP 内核以 **fat binary（gfx1030 + gfx1031 + gfx1032）** 形式提供，
   因此整个 RX 6000 系列都运行原生代码。
3. PyTorch 2.12 `+rocm7.15` 链接该运行时 → 在 RDNA2 Windows 上
   `torch.cuda.is_available() == True`。
4. vLLM 插件 `vllm_windows_rocm` 注册优化内核：用于量化线性层的原生 HIP W4
   GEMV、用于稠密层（包括巨大的共享 lm_head）的 M=1 瘦身 GEMV，且兼容
   CUDA graphs（注册为真正的 torch 算子）。
5. `INSTALL.bat`（引擎：`INSTALL.ps1`，清单：`MANIFEST.json`）从 GitHub
   Releases 下载 4 个预编译归档、从 HuggingFace 下载模型、安装基础 Python、
   修复 venv、并通过基准测试验证。幂等：只下载缺失的部分。

## 完整性校验（SHA256）

每个下载的分片在解压前都会与同一 GitHub release 上发布的
**`SHA256SUMS.txt`** 进行比对：

- 损坏或不完整的下载会被标记为 `checksum mismatch` 并拒绝 —— 重新运行
  `INSTALL.bat`，它会自动续传
- 失败的下载绝不会留下残缺文件（会删除并重试一次）
- 没有校验文件的 fork/镜像：传入 `-SkipChecksum`

`scripts\make_checksums.ps1` 重新生成校验文件；`scripts\validate_release.ps1`
将清单与 release 比对（两者均在 CI 中运行）。见
[docs/RELEASING.md](docs/RELEASING.md) 与 [SECURITY.md](SECURITY.md)。

## 安装后的目录布局

```
C:\Python311                                 Python 3.11.9
C:\TheRock\.venv                             torch 2.12+rocm7.15 venv（vLLM 0.19.1）
C:\TheRock\build\dist\rocm                   ROCm 运行时库
C:\TheRock\ROCM_VLLM_RUNTIME                 vLLM + 插件 + rocBLAS + rocblas-bench
C:\vw_cext_build, C:\vw_hipgemv_build        原生 HIP 内核
%USERPROFILE%\.cache\huggingface             模型权重
```

发布归档（本仓库 **Releases** 页面，标签 `V2.1`）：

| 归档 | 大小 | 内容 |
|---|---:|---|
| `the-rock-venv.tar.zst` | 1.34 GB | torch ROCm venv |
| `therock-rocm-dist.tar.zst` | 0.85 GB | ROCm 运行时 |
| `vllm-stack.tar.zst` | 0.14 GB | vLLM + 插件 + rocBLAS + rocblas-bench.exe |
| `native-kernels.tar.zst` | 约 1 MB | HIP GEMV 内核（fat binary） |

## 仓库结构

```
vllm-rocm-windows-rdna2-oneclick/
├── INSTALL.bat               一键安装程序（入口）
├── INSTALL.ps1               安装引擎（下载、SHA256 校验、解压）
├── UNINSTALL.bat             彻底卸载已安装环境
├── UNINSTALL.ps1             卸载引擎
├── CHAT.bat                  打开网页聊天（自动启动服务端）
├── SERVE.bat                 单独启动模型服务端（OpenAI API）
├── VERIFY.bat                一次运行全部 3 项验证
├── chat.html                 AI Studio 风格聊天界面（Thinking 动画 + tok/s + 运行参数）
├── MANIFEST.json             release 资源清单，每个显卡家族一个 variant
├── SECURITY.md               漏洞报告 + 供应链说明
├── CONTRIBUTING.md           如何验证显卡、报告 Bug、参与贡献
├── plugin_overrides/         调优插件模块（awq_gemv、bf16_gemv）
├── kernels/                  原生 HIP W4 GEMV 源码 + 按家族构建脚本
├── scripts/                  模型服务端、基准测试、校验、release 验证
├── docs/                     多架构 + 发布指南
├── assets/                   验证截图
└── results/                  原始基准日志 + 优化曲线图
```

## 更换模型

编辑 `config.bat`（由安装程序生成）：将 `SERVED_MODEL` 设为模型文件夹路径、
`MODEL_NAME` 设为聊天/API 中显示的名称，然后重新运行 `CHAT.bat`。基准测试使用
`BENCH_MODEL`（同一文件）：

```bat
C:\TheRock\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('<owner>/<repo>'))"
```

打印出的快照文件夹路径即填入 `config.bat` 的值。

### 安装时直接指定模型

`INSTALL.bat` 接受模型 ID，无需手动编辑 `config.bat`：

```bat
INSTALL.bat                        默认：cyankiwi/Qwen3.5-4B-AWQ-4bit（4 GB 显卡自动使用 Qwen/Qwen2.5-1.5B-Instruct-AWQ）
INSTALL.bat -Model <owner>/<model>  任意公开的 AWQ/GPTQ 模型
INSTALL.bat someuser                从你自己的 fork 的 releases 下载
INSTALL.bat -Variant rdna3          强制指定显卡家族（rdna2/rdna3/rdna4）
```

或直接使用引擎（任意公开的 AWQ/GPTQ 模型）：

```bat
powershell -ExecutionPolicy Bypass -File INSTALL.ps1 -Model <owner>/<model>
```

### 12 GB 显存推荐模型（AWQ 4-bit）

| 模型 | 显存（AWQ 4-bit） | 预期 tok/s* | 备注 |
|---|---:|---:|---|
| Qwen3.5-4B AWQ（`cyankiwi/...`） | 约 3.5 GB | **59-62** | 已验证（默认） |
| Gemma 3 4B AWQ（`gaunernst/gemma-3-4b-it-int4-awq`） | 约 3.5 GB | 约 55-60 | 稳妥的中型备选 |
| Llama 3.2 3B AWQ | 约 2.5 GB | 约 60-65 | 轻量 |

*_同为 RX 6750 XT。更大的上下文窗口会按比例降低 tok/s。
注意：Qwen3.5 的 8B 级模型是 9B 稠密模型 —— 目前还没有为其发布 AWQ/GPTQ
量化版本（只有 MLX/GGUF），因此这里可使用任意公开的 AWQ 4-bit 模型；
本方案并不依赖 Qwen。_

## 故障排查

| 症状 | 解决办法 |
|---|---|
| `No module named 'vllm._C'` 警告 | 正常现象 —— Windows 插件改用原生内核 |
| `checksum mismatch for ...` | 下载损坏 —— 重新运行 `INSTALL.bat`，它会重新获取损坏的分片 |
| `download failed: ...rdna3-...` | 该显卡家族为实验性 —— 归档尚未发布（见 [docs/MULTIARCH.md](docs/MULTIARCH.md)） |
| Python 安装器退出码 1601 | 自动 NuGet 回退已接管；无需操作 |
| 解压 "Can't unlink" 错误 | 关闭残留的 Python 进程，重新运行 `INSTALL.bat`（可续传） |
| tok/s 偏低 | 关闭其他 GPU 负载；冷卡上确认 `overall_tok_s` ≥ 55 |
| 非 RDNA2 显卡 | 安装程序检测家族（RDNA3/4 实验性），未知显卡给出警告 |
| `tar` 对 `.zst` 文件提示 "Unrecognized archive format" | 老版本 Windows 10 缺少 zstd —— 安装 Windows 更新（bsdtar 3.5+ 自带 libzstd；Windows 11 无此问题） |
| 聊天页面显示 "Server error" | 模型仍在加载 —— 等待 SERVE 窗口出现 "Application startup complete"（**首次 8-10 分钟** —— 内核编译；之后约 1 分钟） |
| `ERROR: SERVED_MODEL is not set` | `config.bat` 缺失（它由 `INSTALL.bat` 生成，且有意加入 gitignore）。运行 `INSTALL.bat`，或从旧安装复制 `config.bat`，或手动创建：`set "SERVED_MODEL=C:\path\to\model\folder"` + `set "MODEL_NAME=MyModel"` |
| 回答中出现 "Thinking Process…" | 那是模型在回答内打印的推理过程 —— 在 `config.bat` 中设置 `THINKING=0` 获得直接回答（然后重新运行 `CHAT.bat`） |
| 自定义调试 | `set BENCH_MODE=eager`、`set VLLM_WIN_HIPGEMV=0`、`set VLLM_WIN_BF16_GEMV=0` |

## 重新编译内核

`kernels\build_kernels.ps1` 按显卡家族重新编译 `kernels/src/gemv_w4.cu`
（gfx1030+1031+1032 的 fat binary，或 RDNA3/4 的 gfx1100/1200；需要 HIP SDK
+ MSVC）。仅用于非 RDNA2 目标或特殊 torch ABI；随附的 rdna2 fat binary
已覆盖全部 RDNA2。

## 致谢

基于 ROCm/TheRock、PyTorch ROCm 与 vLLM 项目构建。与 AMD 无关。

## 许可证

[Apache License 2.0](LICENSE)