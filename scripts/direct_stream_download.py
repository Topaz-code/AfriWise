"""
Direct Fast & Resumable Stream Downloader for Phi-3 Mini Shard 1.

Downloads directly from Hugging Face CDN using requests with HTTP Range headers,
visual progress logging, and automatic snapshot linking.
"""
import os
import sys
import time
import requests
from pathlib import Path

HF_HUB_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--microsoft--Phi-3-mini-4k-instruct"
BLOBS_DIR = HF_HUB_DIR / "blobs"
SNAPSHOTS_DIR = HF_HUB_DIR / "snapshots" / "f39ac1d28e925b323eae81227eaba4464caced4e"

URL = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/model-00001-of-00002.safetensors"
TARGET_FILENAME = "model-00001-of-00002.safetensors"

def download_shard():
    BLOBS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if incomplete file exists
    incomplete_files = list(BLOBS_DIR.glob("b7492726*.incomplete"))
    if incomplete_files:
        blob_path = incomplete_files[0]
    else:
        blob_path = BLOBS_DIR / "b7492726c01287bf6e13c3d74c65ade3d436d50da1cf5bb6925bc962419d6610.incomplete"

    dest_blob = BLOBS_DIR / "b7492726c01287bf6e13c3d74c65ade3d436d50da1cf5bb6925bc962419d6610"
    snapshot_link = SNAPSHOTS_DIR / TARGET_FILENAME

    if dest_blob.exists() and dest_blob.stat().st_size > 4 * 1024**3:
        print(f"[OK] {TARGET_FILENAME} already fully downloaded: {dest_blob.stat().st_size / (1024**2):.1f} MB")
        if not snapshot_link.exists():
            import shutil
            # On Windows without symlink privileges, create hardlink or copy
            try:
                os.link(str(dest_blob), str(snapshot_link))
            except Exception:
                shutil.copyfile(str(dest_blob), str(snapshot_link))
        return

    existing_size = blob_path.stat().st_size if blob_path.exists() else 0
    print(f"Starting/Resuming download from byte {existing_size} ({existing_size / (1024**2):.1f} MB)...")

    headers = {"User-Agent": "AfriWise-Downloader/1.0"}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    resp = requests.get(URL, headers=headers, stream=True, timeout=30)
    
    if resp.status_code not in (200, 206):
        print(f"Error: Server returned status {resp.status_code}")
        # If range request failed, start fresh
        if resp.status_code == 416:
            headers.pop("Range", None)
            existing_size = 0
            resp = requests.get(URL, headers=headers, stream=True, timeout=30)

    total_bytes = int(resp.headers.get("content-length", 0)) + existing_size
    mode = "ab" if existing_size > 0 else "wb"

    downloaded = existing_size
    last_print = time.time()
    t0 = time.time()
    chunk_size = 1024 * 1024 * 4  # 4 MB chunks

    with open(blob_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if time.time() - last_print > 10:
                    pct = (downloaded / total_bytes * 100) if total_bytes > 0 else 0
                    speed = (downloaded - existing_size) / (time.time() - t0) / (1024**2)
                    print(f"  [Progress] {downloaded / (1024**2):.1f} / {total_bytes / (1024**2):.1f} MB ({pct:.1f}%) | {speed:.2f} MB/s")
                    last_print = time.time()

    print(f"\nDownload finished: {downloaded / (1024**2):.1f} MB")
    
    # Rename incomplete to final blob
    if blob_path.exists():
        if dest_blob.exists():
            dest_blob.unlink()
        blob_path.rename(dest_blob)
        print(f"[OK] Renamed to {dest_blob.name}")

    # Create link in snapshot
    if snapshot_link.exists():
        snapshot_link.unlink()
    try:
        os.link(str(dest_blob), str(snapshot_link))
        print(f"[OK] Hardlink created: {snapshot_link}")
    except Exception as e:
        import shutil
        print(f"[Fallback] Copying to snapshot: {e}")
        shutil.copyfile(str(dest_blob), str(snapshot_link))

    print(f"🎉 {TARGET_FILENAME} ready in snapshot directory!")

if __name__ == "__main__":
    download_shard()
