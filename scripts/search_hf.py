"""
search_hf.py — Discover all HuggingFace datasets for African languages.
"""
import io
import json
import sys
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
headers = {"User-Agent": "Mozilla/5.0"}

for q in ["igbo", "efik", "bini", "edo", "ibibio", "masakhane", "yoruba", "hausa"]:
    url = f"https://huggingface.co/api/datasets?search={q}&limit=8"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            ids = [d["id"] for d in data]
            print(f"HF search '{q}': {ids}")
    except Exception as e:
        print(f"Error for {q}: {e}")
