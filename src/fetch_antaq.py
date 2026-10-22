"""Pull only the ANTAQ raw tables we need out of the 5.5 GB Zenodo zip, via HTTP range requests."""
import argparse
import io
import zipfile
from pathlib import Path

import requests

URL = "https://zenodo.org/api/records/20549767/files/antaq-turnaround-data-v1.2.zip/content"
TABLES = ["Atracacao", "TemposAtracacao", "Carga"]
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        self.session = requests.Session()
        self.size = int(self.session.head(url, allow_redirects=True).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        r.raise_for_status()
        n = len(r.content)
        b[:n] = r.content
        self.pos += n
        return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("years", nargs="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    # big buffer so zipfile's small reads become few large range requests
    zf = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=8 << 20))
    raw = [i for i in zf.infolist() if "/raw/" in i.filename and not i.is_dir()]
    if args.list:
        for i in raw:
            print(f"{i.file_size / 1e6:9.1f} MB  {i.filename}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    wanted = {f"{y}{t}.txt" for y in args.years for t in TABLES}
    for i in raw:
        name = Path(i.filename).name
        if name in wanted or "/categories/" in i.filename or "/metadata/" in i.filename:
            dest = OUT / name
            if dest.exists() and dest.stat().st_size == i.file_size:
                print("skip", name)
                continue
            print(f"get  {name} ({i.file_size / 1e6:.0f} MB)", flush=True)
            with zf.open(i) as src, open(dest, "wb") as dst:
                while chunk := src.read(8 << 20):
                    dst.write(chunk)


if __name__ == "__main__":
    main()
