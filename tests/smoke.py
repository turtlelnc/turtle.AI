"""Run real training and check invalid inputs in an isolated directory."""
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


binary = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="turtle-smoke-") as temp:
    root = Path(temp)
    (root / "train.txt").write_text("abc", encoding="utf-8")
    config = {
        "model": {"seq_len": 2, "d_model": 4, "vocab_size": 260},
        "training": {
            "corpus_path": "train.txt", "epochs": 2,
            "save_point": 1, "save_path": "model.bin",
            "seed": 7,
        },
    }

    def run(expected=0, message=""):
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        result = subprocess.run(
            [str(binary), "config.json"], cwd=root, text=True,
            capture_output=True, timeout=30,
        )
        if result.returncode != expected or message not in result.stdout + result.stderr:
            raise RuntimeError(f"Unexpected training result: {result.stdout}{result.stderr}")
        return result.stdout

    output = run(message="Model saved")
    losses = re.findall(r"Loss: ([\d.]+)", output)
    if not losses or not all(math.isfinite(float(loss)) and float(loss) > 0 for loss in losses):
        raise RuntimeError("Training did not report a finite positive loss")
    if (root / "model.bin").stat().st_size == 0:
        raise RuntimeError("Training did not write a checkpoint")

    def unpack_model(path):
        data = path.read_bytes()
        if data[:8] != b"TRTLMODL":
            raise RuntimeError("Checkpoint does not have a versioned header")
        length = struct.unpack("<I", data[8:12])[0]
        return json.loads(data[12:12 + length]), data[12 + length:]

    def pack_model(metadata, payload):
        header = json.dumps(metadata).encode()
        return b"TRTLMODL" + struct.pack("<I", len(header)) + header + payload

    metadata, payload = unpack_model(root / "model.bin")
    norm_bytes = 8 * 4 * 4 * 4  # blocks * (two norms, gamma+beta) * width * float size
    norm_values = struct.unpack("=" + "f" * (norm_bytes // 4), payload[-norm_bytes:])
    if all(value == (1.0 if (i // 4) % 2 == 0 else 0.0) for i, value in enumerate(norm_values)):
        raise RuntimeError("Training did not update LayerNorm parameters")
    run(message="Loading weights")
    valid_checkpoint = (root / "model.bin").read_bytes()
    metadata, payload = unpack_model(root / "model.bin")
    for key, invalid in {
        "format_version": 999, "d_model": 5, "seq_len": 3,
        "vocab_size": 261, "block_count": 9,
        "tokenizer_fingerprint": "wrong", "byte_order": "wrong",
    }.items():
        bad = dict(metadata, **{key: invalid})
        invalid_checkpoint = pack_model(bad, payload)
        (root / "model.bin").write_bytes(invalid_checkpoint)
        run(1, f"Checkpoint metadata mismatch: {key}")
        if (root / "model.bin").read_bytes() != invalid_checkpoint:
            raise RuntimeError("Rejected checkpoint was overwritten")
    (root / "model.bin").write_bytes(pack_model(metadata, struct.pack("=f", math.nan) + payload[4:]))
    run(1, "Checkpoint contains NaN/Inf")
    (root / "model.bin").write_bytes(valid_checkpoint + b"extra")
    run(1, "Checkpoint is truncated or incompatible")
    (root / "model.bin").write_bytes(b"TRTLMODL" + struct.pack("<I", 2**32 - 1))
    run(1, "Checkpoint header length is invalid")
    (root / "model.bin").write_bytes(payload[:-norm_bytes])
    run(message="Legacy checkpoint")
    unpack_model(root / "model.bin")  # Legacy saves are upgraded automatically.
    (root / "model.bin").write_bytes(b"bad")
    run(1, "Checkpoint is truncated")
    (root / "model.bin").unlink()
    config["model"]["seq_len"] = 0
    run(1, "must be positive")
    config["model"]["seq_len"] = 4
    run(1, "seq_len + 1")
    config["model"]["seq_len"] = 2
    config["training"]["corpus_path"] = "missing.txt"
    run(1, "Corpus does not exist")
    config["training"]["corpus_path"] = "train.txt"
    config["training"]["save_path"] = "missing/model.bin"
    run(1, "Cannot open checkpoint for writing")
    (root / "config.json").write_text("{", encoding="utf-8")
    result = subprocess.run([str(binary)], cwd=root, capture_output=True, text=True, timeout=30)
    if result.returncode != 1 or "Error:" not in result.stderr:
        raise RuntimeError("Invalid JSON did not produce a clear error")

    # Text training leaves vision weights at initialization. With an identical
    # seed, VLM training must update those same weights through the image path.
    (root / "train.txt").write_text("abc " * 100, encoding="utf-8")
    config["model"]["seq_len"] = 200
    config["training"].update(save_path="text-reference.bin", epochs=1, learning_rate=0.01)
    run()
    _, text_payload = unpack_model(root / "text-reference.bin")
    offset = 260 * 4 * 4  # embedding weights
    vision_bytes = (768 * 4 + 4) * 4
    initial_vision = text_payload[offset:offset + vision_bytes]
    dataset = root / "vlm"
    dataset.mkdir()
    (dataset / "train.json").write_text(json.dumps([{"answer": "abc"}]), encoding="utf-8")
    # stb_image detects the content even though the existing dataset convention uses .jpg.
    (dataset / "train_0.jpg").write_bytes(b"P6\n1 1\n255\n" + bytes([127, 64, 32]))
    config["training"].update(corpus_path="vlm", save_path="vlm.bin")
    output = run(message="Starting VLM")
    if not re.search(r"Loss: [1-9\d]", output):
        raise RuntimeError("VLM training did not supervise answer tokens")
    _, vlm_payload = unpack_model(root / "vlm.bin")
    if initial_vision == vlm_payload[offset:offset + vision_bytes]:
        raise RuntimeError("VLM training did not update vision projection")
    if initial_vision[-16:] == vlm_payload[offset + vision_bytes - 16:offset + vision_bytes]:
        raise RuntimeError("VLM training did not update vision bias")
    run(message="Loading weights")
    (dataset / "train_0.jpg").unlink()
    run(1, "Cannot load image")
    (dataset / "train.json").write_text('[{"answer": ""}]', encoding="utf-8")
    run(1, "nonempty answer string")
print("Training smoke tests passed")
