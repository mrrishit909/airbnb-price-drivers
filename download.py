"""Step 1: the Inside Airbnb Austin files of 22 June 2026: three files in three formats.

    ./venv/bin/python download.py   -> data/ (3 files), results/pull.json

Inside Airbnb (insideairbnb.com/get-the-data), "Austin, Texas, United States, 22 June 2026", licensed under Creative Commons Attribution 4.0. Public listing information collected from airbnb.com. The files are
listings.csv.gz (the detailed listings, gzipped CSV), neighbourhoods.csv (the zip codes Inside Airbnb uses as neighbourhoods) and neighbourhoods.geojson (their outlines, GeoJSON). About 8 MB, so none of it is committed;
this script fetches it. Host names, listing titles and descriptions are in the listings file and are never used or published: the results are summaries.
"""
import csv
import gzip
import hashlib
import io
import json
import ssl
import urllib.request
from datetime import date
from pathlib import Path

import certifi

HERE = Path(__file__).parent
BASE = "https://data.insideairbnb.com/united-states/tx/austin/2026-06-22/"
FILES = {"listings.csv.gz": "data/listings.csv.gz", "neighbourhoods.csv": "visualisations/neighbourhoods.csv", "neighbourhoods.geojson": "visualisations/neighbourhoods.geojson"}
CTX = ssl.create_default_context(cafile=certifi.where())


def main():
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    info = {"pulled": date.today().isoformat(), "files": {}}
    for name, path in FILES.items():
        blob = urllib.request.urlopen(urllib.request.Request(BASE + path, headers={"User-Agent": "research-portfolio (Rishit Raj Mathur)"}), context=CTX, timeout=300).read()
        (HERE / "data" / name).write_bytes(blob)
        text = gzip.decompress(blob).decode("utf-8") if name.endswith(".gz") else blob.decode("utf-8")
        rows = len(list(csv.reader(io.StringIO(text)))) - 1 if name.endswith((".csv", ".csv.gz")) else len(json.loads(text)["features"])
        info["files"][name] = {"url": BASE + path, "bytes": len(blob), "rows": rows, "sha256": hashlib.sha256(blob).hexdigest()}
    (HERE / "results" / "pull.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: (v["rows"], v["bytes"]) for k, v in info["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
