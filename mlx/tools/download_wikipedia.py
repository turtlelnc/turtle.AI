"""Download a bounded Wikipedia subset with source IDs and article-level splits.

Only the Python standard library is needed. Generated data stays in build-mlx.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


def fetch_json(url):
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "turtle-ai-training-check/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch_rows(offset, length):
    query = urllib.parse.urlencode({
        "dataset": "wikimedia/wikipedia", "config": "20231101.en",
        "split": "train", "offset": offset, "length": length,
    })
    return fetch_json("https://datasets-server.huggingface.co/rows?" + query)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build-mlx/wikipedia"))
    parser.add_argument("--offset", type=int, default=1000)
    parser.add_argument("--articles", type=int, default=1000)
    parser.add_argument("--validation-every", type=int, default=10)
    args = parser.parse_args()
    if args.offset < 0 or args.articles < 2 or args.validation_every < 2:
        parser.error("offset must be nonnegative; articles and validation-every must be >= 2")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {split: args.output / (split + ".txt") for split in ("train", "validation")}
    if any(path.exists() for path in paths.values()):
        parser.error("output files already exist; choose a new directory")
    source = fetch_json("https://huggingface.co/api/datasets/wikimedia/wikipedia")
    manifest = {
        "dataset": "wikimedia/wikipedia", "config": "20231101.en", "split": "train",
        "source": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "source_revision": source["sha"],
        "license": "CC BY-SA 3.0; Wikipedia content also subject to GFDL",
        "offset": args.offset, "requested_articles": args.articles,
        "selection": "contiguous Dataset Viewer rows, not a random or representative sample",
        "validation_every": args.validation_every, "articles": [],
    }
    streams = {split: path.open("wb") for split, path in paths.items()}
    completed = False
    try:
        for start in range(args.offset, args.offset + args.articles, 25):
            length = min(25, args.offset + args.articles - start)
            page = fetch_rows(start, length)
            rows = page["rows"]
            if len(rows) != length:
                raise RuntimeError("Dataset Viewer returned fewer rows than requested")
            for item in rows:
                if item.get("truncated_cells"):
                    raise RuntimeError(f"Truncated Dataset Viewer row: {item['row_idx']}")
                row = item["row"]
                if not row["text"].strip():
                    raise RuntimeError(f"Empty article: {row['id']}")
                index = item["row_idx"] - args.offset
                split = "validation" if index % args.validation_every == 0 else "train"
                payload = (row["title"] + "\n" + row["text"].rstrip() + "\n\n").encode("utf-8")
                streams[split].write(payload)
                manifest["articles"].append({
                    "row": item["row_idx"], "id": row["id"], "title": row["title"],
                    "url": row["url"], "split": split, "bytes": len(payload),
                })
            print(f"Downloaded {len(manifest['articles'])}/{args.articles} articles", flush=True)
        completed = True
    finally:
        for stream in streams.values():
            stream.close()
        if not completed:
            for path in paths.values():
                path.unlink(missing_ok=True)
    manifest["files"] = {
        split: {"bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "articles": sum(item["split"] == split for item in manifest["articles"])}
        for split, path in paths.items()
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["files"], indent=2))


if __name__ == "__main__":
    main()
