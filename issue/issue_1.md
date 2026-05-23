# [task] 搭建 DFlash SGLang 测试环境并预缓存评测模型

## [Task] 核心任务目标

**执行动作**：基于 DFlash 项目 README 文档，搭建专用于 SGLang 后端的 Python 虚拟环境，使用 `uv pip` 安装依赖；同时预下载并缓存评测所需的基座模型与草稿模型，确保后续可通过 `sglang.launch_server` 正常启动服务。

### 预期交付物
- 一份环境搭建验证报告（Markdown 格式），存入 `docs/setup_dflash_sglang_env.md`
- 内容应包含：
  - 虚拟环境创建命令与版本信息
  - 依赖安装步骤及关键包版本（sglang、torch、flash-attn 等）
  - 模型下载缓存路径及大小
  - 最小化启动验证命令及输出日志片段

### 探索维度或建议步骤
1. **创建独立虚拟环境**：使用 `uv venv` 创建专用环境（建议路径：`/home/shanchuang/dflash/.venv-sglang`），避免与其他后端冲突
2. **安装 SGLang 依赖**：执行 `uv pip install -e ".[sglang]"`，记录关键依赖版本
3. **预下载模型**：
   - 基座模型：`Qwen/Qwen3.6-27B`
   - DFlash 草稿模型：`z-lab/Qwen3.6-27B-DFlash`
   - 若存在 eagle3 相关草稿模型，一并下载
4. **验证模型缓存**：确认模型已缓存至 HuggingFace 缓存目录（默认 `~/.cache/huggingface/hub`）
5. **最小化启动测试**：参考 README 中 SGLang 示例，执行启动命令验证环境可用性（可使用较小参数如 `--max-num-batched-tokens` 进行快速验证）

### 注意事项
- 每个后端（Transformers、SGLang、vLLM、MLX）需独立虚拟环境，严禁混用
- SGLang 启动前需设置环境变量 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`
- 若需启用实验性调度重叠功能，需额外设置 `SGLANG_ENABLE_SPEC_V2=1` 等变量
- 模型下载需确认网络可达 HuggingFace，必要时配置镜像站或代理
- 使用 `huggingface-cli download` 或 `snapshot_download` 预缓存模型，避免首次启动时等待
- 验证启动时注意 GPU 显存占用，Qwen3.6-27B 需确保显存充足

【项目全局规范 (重要！请务必遵守其中的路径与隔离规范，严禁幻觉不存在的目录)】
- 项目根目录：`/home/shanchuang/dflash`
- 虚拟环境建议路径：`/home/shanchuang/dflash/.venv-sglang`
- 模型缓存路径：`~/.cache/huggingface/hub` 或自定义路径
- 交付文档路径：`/home/shanchuang/dflash/docs/setup_dflash_sglang_env.md`

【上下文参考树】
- README.md 安装章节：明确 SGLang 安装命令 `uv pip install -e ".[sglang]"`
- README.md 支持模型表格：Qwen3.6-27B 对应 DFlash 草稿模型 `z-lab/Qwen3.6-27B-DFlash`
- README.md SGLang Quick Start：提供完整启动参数示例

【具体代码详情片段】
详见附件 README.md 内容$

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: Tensor Parallel (TP) = 4
- **Interconnect**: NVLink

> ⚠️ 注意：所有评估的启动指令和执行脚本必须严格遵循 `tp-size=4` 等效并行度，并且在模型加载和执行期间严密监控 VRAM 占用（单卡最高不能超过16GB）。$

### 📦 补充交付产物要求 (Deliverables)
- 除了最终的测试结果或环境配置结果外，**必须**将本次任务涉及到的**可复现一键执行脚本**（如 `setup_env.sh`、`run_baseline.sh`、`run_eagle3.sh`、`run_dflash.sh` 等）一并作为交付产物编写并沉淀到本地代码库中。

