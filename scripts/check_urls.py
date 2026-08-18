"""
check_urls.py — Probe large-scale corpus download sources.
"""
import urllib.request

urls = {
    "JW300 Igbo": "https://object.pouta.csc.fi/OPUS-JW300/v1/mono/ig.txt.gz",
    "JW300 Efik": "https://object.pouta.csc.fi/OPUS-JW300/v1/mono/efi.txt.gz",
    "JW300 Bini": "https://object.pouta.csc.fi/OPUS-JW300/v1/mono/bin.txt.gz",
    "eBible Igbo": "https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/ibo-ibo.txt",
    "OPUS Books Igbo": "https://object.pouta.csc.fi/OPUS-Books/v1/mono/ig.txt.gz"
}

for name, url in urls.items():
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            cl = resp.headers.get("Content-Length", "unknown")
            print(f"[OK] {name}: {resp.status} (Size: {int(cl)/1024/1024:.2f} MB)" if cl != "unknown" else f"[OK] {name}: {resp.status}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
