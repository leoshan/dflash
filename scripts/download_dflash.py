import os
from huggingface_hub import snapshot_download

def download_model():
    repo_id = "z-lab/Qwen3.6-27B-DFlash"
    token = os.getenv("HF_TOKEN")
    
    print(f"开始下载模型: {repo_id} ...")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"下载完成! 模型存储路径: {path}")
    except Exception as e:
        print(f"下载失败: {e}")

if __name__ == "__main__":
    download_model()
