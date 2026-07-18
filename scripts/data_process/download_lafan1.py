import os
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# 1. 配置根目录（改成你自己的绝对路径）
PROJECT_ROOT = Path(__file__).resolve().parent  # 脚本所在目录
DATA_ROOT = PROJECT_ROOT / "datasets" / "lafan1"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
DOWNLOAD_DIR = DATA_ROOT / "downloads"

# 2. 官方 LAFAN1 链接（Ubisoft GitHub 提供的 zip）
LAFAN1_URL = "https://github.com/ubisoft/ubisoft-laforge-animation-dataset/archive/refs/heads/master.zip"
LAFAN1_ZIP = DOWNLOAD_DIR / "lafan1_master.zip"

def ensure_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, DOWNLOAD_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def download_lafan1():
    if LAFAN1_ZIP.exists():
        print(f"[INFO] Zip already exists: {LAFAN1_ZIP}")
    else:
        print(f"[INFO] Downloading LAFAN1 from {LAFAN1_URL}")
        urlretrieve(LAFAN1_URL, LAFAN1_ZIP)
        print(f"[INFO] Saved to {LAFAN1_ZIP}")

def extract_lafan1():
    print(f"[INFO] Extracting {LAFAN1_ZIP} to {RAW_DIR}")
    with zipfile.ZipFile(LAFAN1_ZIP, "r") as zf:
        # 解压到临时目录
        zf.extractall(DOWNLOAD_DIR)

    # GitHub zip 结构类似：ubisoft-laforge-animation-dataset-master/lafan1/
    root_candidates = list(DOWNLOAD_DIR.glob("ubisoft-laforge-animation-dataset-*"))
    if not root_candidates:
        raise RuntimeError("Cannot find extracted ubisoft-laforge-animation-dataset-* directory")

    src_root = root_candidates[0] / "lafan1"
    if not src_root.exists():
        raise RuntimeError(f"Cannot find 'lafan1' dir under {src_root.parent}")

    # 把所有 BVH 拷贝到 RAW_DIR（保持子目录结构）
    for path in src_root.rglob("*.bvh"):
        rel = path.relative_to(src_root)
        dst = RAW_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())

    print(f"[INFO] Copied BVH files to {RAW_DIR}")

def list_bvh():
    bvh_files = sorted(RAW_DIR.rglob("*.bvh"))
    print(f"[INFO] Found {len(bvh_files)} BVH files under {RAW_DIR}")
    for f in bvh_files[:10]:
        print("   ", f)

if __name__ == "__main__":
    ensure_dirs()
    download_lafan1()
    extract_lafan1()
    list_bvh()
    print("[INFO] Done.")
