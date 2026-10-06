"""Run a small CPU training, checkpoint-resume, and generation experiment.

Run download_wikipedia.py first. Artifacts and full logs remain in build-mlx.
Training loss is a functionality check, not a held-out quality evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time


def checkpoint_info(path):
    with path.open("rb") as stream:
        if stream.read(8) != b"TRTLMX01":
            raise RuntimeError("Missing MLX checkpoint header")
        length = struct.unpack("<I", stream.read(4))[0]
        metadata = json.loads(stream.read(length))
        step = struct.unpack("=i", stream.read(4))[0]
        # GLR2 stores native layout fields. The current MLX model weights are FP16.
        magic, enabled, rank, minimum = struct.unpack("=IiiQ", stream.read(20))
        if magic != 0x474C5232:
            raise RuntimeError("Expected GLR2 optimizer payload")
        count = struct.unpack("=Q", stream.read(8))[0]
        weights_hash = hashlib.sha256()
        weight_bytes = 0
        for _ in range(count):
            size = struct.unpack("=Q", stream.read(8))[0]
            weights = stream.read(size)
            if len(weights) != size:
                raise RuntimeError("Truncated model weights")
            weight_bytes += size
            weights_hash.update(weights)
            packed = struct.unpack("=Q", stream.read(8))[0]
            stream.seek(packed * 4, 1)
        if stream.tell() != path.stat().st_size:
            raise RuntimeError("Unexpected checkpoint payload size")
    return {"metadata": metadata, "completed_step": step, "bytes": path.stat().st_size,
            "parameter_tensors": count, "fp16_parameters": weight_bytes // 2,
            "weights_sha256": weights_hash.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build-mlx/turtle-mlx"))
    parser.add_argument("--data", type=Path, default=Path("build-mlx/wikipedia"))
    parser.add_argument("--output", type=Path, default=Path("build-mlx/wiki-small"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resume-steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps <= 200 or args.resume_steps <= 0:
        parser.error("steps must exceed the 200-step warmup and resume-steps must be positive")
    binary = args.binary.resolve()
    data = args.data.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory already exists; choose a fresh run directory")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        digest = hashlib.sha256((data / (split + ".txt")).read_bytes()).hexdigest()
        if digest != manifest["files"][split]["sha256"]:
            raise RuntimeError(f"Dataset checksum mismatch: {split}")
    output.mkdir(parents=True)
    checkpoint = output / "model.ckpt"
    tokenizer = output / "tokenizer.bpe"
    options = {
        "device": "cpu", "dim": 64, "seq-len": 128, "max-loop": 2,
        "window-size": 64, "global-topk": 8, "experts": 4, "moe-topk": 2,
        "wide-blocks": 1, "batch-size": 4, "vocab-size": 1024,
        "peak-lr": 0.001, "seed": 7, "galore-off": 1,
        "optimizer-memory": "resident", "compile-train-graph": 1,
    }
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    stages = []

    def run(name, mode=None, **changes):
        command = [str(binary)] + (mode or [])
        for key, value in dict(options, **changes).items():
            command += ["--" + key, str(value)]
        log = output / (name + ".log")
        print(f"Starting {name}: {' '.join(command)}", flush=True)
        start = time.monotonic()
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=output, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
        seconds = time.monotonic() - start
        text = log.read_text(encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}):\n{text[-6000:]}")
        if re.search(r"(?<!\w)(nan|[+-]?inf)(?!\w)", text, re.I):
            raise RuntimeError(f"Nonfinite result in {log}")
        stage = {"name": name, "command": command, "seconds": seconds, "log": log.name}
        stages.append(stage)
        print(f"Completed {name} in {seconds:.2f}s", flush=True)
        return text, stage

    common = {"data-dir": data / "train.txt", "model-out": checkpoint,
              "tokenizer-out": tokenizer}
    text, stage = run("train", **common, steps=args.steps)
    first = checkpoint_info(checkpoint)
    if first["completed_step"] != args.steps:
        raise RuntimeError("Initial checkpoint step mismatch")
    shutil.copyfile(checkpoint, output / "before-resume.ckpt")
    losses = [float(x) for x in re.findall(r"任务Loss\([^)]*\)\s+([\d.eE+-]+)", text)]
    if not losses:
        raise RuntimeError("No task-loss measurements were logged")
    stage["sampled_task_losses"] = losses
    stage["first_5_mean"] = sum(losses[:5]) / len(losses[:5])
    stage["last_5_mean"] = sum(losses[-5:]) / len(losses[-5:])
    text, _ = run("resume", **common, steps=args.steps + args.resume_steps)
    final = checkpoint_info(checkpoint)
    if "成功恢复状态" not in text or final["completed_step"] != args.steps + args.resume_steps:
        raise RuntimeError("Checkpoint resume did not advance the completed step")
    if final["weights_sha256"] == first["weights_sha256"]:
        raise RuntimeError("Model weights unchanged after further training")
    for index, prompt in enumerate(("The history of mathematics", "A city is", "The Solar System")):
        text, stage = run(f"generate-{index + 1}", ["gen", prompt],
                          model=checkpoint, tokenizer=tokenizer, **{"max-tokens": 80, "temp": 0.7})
        if "READY\n" not in text or "END\n" not in text:
            raise RuntimeError("Generation protocol was not completed")
        stage["prompt"] = prompt
        raw = (output / stage["log"]).read_bytes()
        protocol = raw.split(b"READY\n", 1)[1].rsplit(b"END\n", 1)[0]
        pieces = re.findall(rb"TOKEN:(.*?)\n(?=TOKEN:|$)", protocol, re.S)
        stage["generated_tokens"] = len(pieces)
        stage["generated_text"] = b"".join(pieces).decode("utf-8", errors="replace")
    text, _ = run("generate-no-kv", ["gen", "The history of mathematics"],
                  model=checkpoint, tokenizer=tokenizer,
                  **{"max-tokens": 80, "temp": 0.7, "kv-cache": 0})
    if "END\n" not in text:
        raise RuntimeError("Generation without KV cache did not complete")
    evaluations = {}
    for name, untrained in (("eval-baseline", 1), ("eval-trained", 0)):
        text, stage = run(name, ["eval"], model=checkpoint, tokenizer=tokenizer,
                          **{"eval-data": data / "validation.txt", "eval-samples": 64,
                             "eval-untrained": untrained})
        stage["metrics"] = json.loads(text.split("EVAL ", 1)[1])
        evaluations[name] = stage["metrics"]
    summary = {
        "dataset": {key: value for key, value in manifest.items() if key != "articles"},
        "model_options": options, "stages": stages,
        "initial_checkpoint": first, "final_checkpoint": final,
        "validation": evaluations,
        "resume": "Verifies continued updates; changing --steps changes the LR schedule, and RNG state is not restored",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"All stages passed. Results: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
