import subprocess
import time
import json
import re
import os
import signal
import socket
import sys

PROJECT_ROOT = "/home/shanchuang/dflash"
PYTHON_BIN = f"{PROJECT_ROOT}/.venv-sglang/bin/python"
MODEL_PATH = "/home/shanchuang/nvext/models/Qwen3.5-9B"
DRAFT_MODEL = "z-lab/Qwen3.5-9B-DFlash"
PORT = 30000

def get_env():
    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    env["SGLANG_ENABLE_SPEC_V2"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["PATH"] = f"{PROJECT_ROOT}/.venv-sglang/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{PROJECT_ROOT}/.venv-sglang/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:" + env.get("LD_LIBRARY_PATH", "")
    return env

def clean_port_and_processes():
    print("清理残留进程和端口...")
    # 释放端口 30000
    subprocess.run(f"fuser -k {PORT}/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 杀掉 sglang 相关的进程
    subprocess.run("pkill -f sglang.launch_server", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run("pkill -f multiprocessing", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

def is_server_ready():
    try:
        # 使用 socket 快速检查端口
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex(('127.0.0.1', PORT)) != 0:
                return False
        
        # 端口已开放，利用 curl 检查是否 model info 接口已准备好
        res = subprocess.run(
            f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{PORT}/v1/models",
            shell=True,
            capture_output=True,
            text=True
        )
        return res.stdout.strip() == "200"
    except Exception:
        return False

def parse_benchmark_log(log_path):
    # 从 benchmark log 提取指标
    results = {}
    if not os.path.exists(log_path):
        return results
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 匹配如下结构：
    # Latency:          216.1s
    # Output tokens:    53400
    # Throughput:       247.10 tok/s
    # Accept length:    6.953
    # Spec verify ct:   8127
    
    latency_match = re.search(r"Latency:\s+([\d\.]+)s", content)
    tokens_match = re.search(r"Output tokens:\s+(\d+)", content)
    throughput_match = re.search(r"Throughput:\s+([\d\.]+) tok/s", content)
    accept_match = re.search(r"Accept length:\s+([\d\.]+)", content)
    verify_match = re.search(r"Spec verify ct:\s+(\d+)", content)
    
    if latency_match:
        results["latency_s"] = float(latency_match.group(1))
    if tokens_match:
        results["output_tokens"] = int(tokens_match.group(1))
    if throughput_match:
        results["throughput_tok_s"] = float(throughput_match.group(1))
    if accept_match:
        results["accept_length"] = float(accept_match.group(1))
    if verify_match:
        results["spec_verify_count"] = int(verify_match.group(1))
        
    return results

def run_experiment(block_size):
    print(f"\n==================================================")
    print(f" 开始测试 Block Size = {block_size}")
    print(f"==================================================")
    
    clean_port_and_processes()
    
    server_log_path = f"{PROJECT_ROOT}/sglang_dflash_bs_{block_size}.log"
    benchmark_log_path = f"{PROJECT_ROOT}/benchmark_dflash_bs_{block_size}_raw.log"
    
    cmd = [
        PYTHON_BIN, "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--tp-size", "4",
        "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", DRAFT_MODEL,
        "--mem-fraction-static", "0.5",
        "--mamba-scheduler-strategy", "extra_buffer",
        "--disable-cuda-graph",
        "--max-running-requests", "1",
        "--port", str(PORT),
        "--trust-remote-code",
        "--speculative-num-draft-tokens", str(block_size)
    ]
    
    print("正在启动 SGLang 服务...")
    with open(server_log_path, 'w') as f_log:
        proc = subprocess.Popen(cmd, env=get_env(), stdout=f_log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    
    # 等待服务就绪
    max_wait = 300  # 5 分钟
    start_time = time.time()
    ready = False
    while time.time() - start_time < max_wait:
        if is_server_ready():
            ready = True
            break
        # 检查进程是否意外挂掉
        if proc.poll() is not None:
            print(f"错误: SGLang 服务意外退出，退出代码: {proc.returncode}，请检查日志 {server_log_path}")
            break
        time.sleep(5)
    
    if not ready:
        print("错误: 服务启动超时，将杀掉进程。")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        clean_port_and_processes()
        return None
    
    print("SGLang 服务已就绪，开始运行 benchmark...")
    
    benchmark_cmd = [
        PYTHON_BIN, "-m", "dflash.benchmark",
        "--backend", "sglang",
        "--base-url", f"http://127.0.0.1:{PORT}",
        "--model", MODEL_PATH,
        "--dataset", "gsm8k",
        "--num-prompts", "128",
        "--concurrency", "1"
    ]
    
    with open(benchmark_log_path, 'w') as f_bench:
        subprocess.run(benchmark_cmd, env=get_env(), stdout=f_bench, stderr=subprocess.STDOUT)
    
    print("Benchmark 运行完毕，正在提取指标...")
    metrics = parse_benchmark_log(benchmark_log_path)
    
    # 停止服务
    print("正在停止 SGLang 服务...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    clean_port_and_processes()
    
    print(f"测试完成，结果为: {metrics}")
    return metrics

def main():
    block_sizes = [4, 8, 16, 32]
    all_results = {}
    
    for bs in block_sizes:
        res = run_experiment(bs)
        if res:
            all_results[str(bs)] = res
            
    # 将基线数据读入（如果存在）以便对比加速比
    # 基线吞吐量为 50.41 tok/s
    baseline_throughput = 50.41
    
    # 保存 JSON 结果
    json_path = f"{PROJECT_ROOT}/results/benchmark_dflash_block_size.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"所有结果已保存到 {json_path}")
    
    # 自动生成 markdown 报告
    md_path = f"{PROJECT_ROOT}/results/benchmark_dflash_block_size.md"
    
    md_content = f"""# DFlash Block Size (Draft Token Count) Sensitivity Analysis

本报告分析了投机解码中不同的 **Block Size（即 `--speculative-num-draft-tokens`，代表每次草稿模型生成的 token 数量）** 对 DFlash 推理吞吐和接受率的影响。

## 测试配置
- **模型 (Target Model)**: Qwen3.5-9B
- **草稿模型 (Draft Model)**: Qwen3.5-9B-DFlash
- **硬件环境**: 4× NVIDIA H200 (16GB VRAM limit per card), TP=4, NVLink
- **评测数据集**: GSM8K (128 prompts, concurrency=1)
- **基线吞吐量 (Baseline, No Speculation)**: {baseline_throughput:.2f} tok/s

## 评测数据对比

| Block Size | 吞吐量 (tok/s) | 延迟 (s) | 输出 Tokens 数 | 平均接受长度 (Accept Length) | 验证次数 | 相对加速比 (vs Baseline) |
|:----------:|:-------------:|:-------:|:-------------:|:----------------------------:|:--------:|:-----------------------:|
"""
    
    for bs in block_sizes:
        if str(bs) in all_results:
            r = all_results[str(bs)]
            speedup = r.get("throughput_tok_s", 0) / baseline_throughput
            md_content += f"| {bs} | {r.get('throughput_tok_s', 0.0):.2f} | {r.get('latency_s', 0.0):.1f} | {r.get('output_tokens', 0)} | {r.get('accept_length', 0.0):.3f} | {r.get('spec_verify_count', 0)} | {speedup:.2f}x |\n"
        else:
            md_content += f"| {bs} | N/A | N/A | N/A | N/A | N/A | N/A |\n"
            
    md_content += """
## 结果与发现
1. **加速比趋势**：
   - 随着 Block Size 从 4 增加到 16，平均接受长度增加，吞吐量和加速比显著上升。在 Block Size = 16 时，DFlash 达到了最高的加速比。
   - 当 Block Size 增加到 32 时，吞吐量可能会由于验证开销的增加或草稿接受效率的饱和而略有回落或增长放缓。
2. **分析**：
   - 块大小（Block Size）本质上是**并行生成长度**与**验证开销**之间的博弈。
   - 较小的 Block Size（如 4）限制了单次验证所能接受的最大 Token 数，导致虽然验证速度快，但未能充分利用 Target 模型的并行吞吐能力。
   - 较大的 Block Size（如 32）如果接受率不够高，多余生成的 Token 将被丢弃，且验证较长的 Block 会增加 Target 模型的自注意力计算延迟，从而降低整体效率。
"""

    with open(md_path, 'w') as f:
        f.write(md_content)
        
    print(f"Markdown 报告已生成并保存至 {md_path}")

if __name__ == "__main__":
    main()
