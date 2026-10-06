"""Exercise real MLX CPU training, resume, generation, and ELF in isolation."""
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="turtle-mlx-smoke-") as temp:
    root = Path(temp)
    corpus = root / "data"
    corpus.mkdir()
    (corpus / "code.cpp").write_text("int main() { return 1; }\n", encoding="utf-8")
    defaults = {
        "device": "cpu", "dim": 16, "seq-len": 8, "max-loop": 2,
        "window-size": 4, "global-topk": 2, "experts": 2, "moe-topk": 1,
        "wide-blocks": 1, "batch-size": 2, "steps": 2, "vocab-size": 320,
        "galore-off": 1, "optimizer-memory": "resident",
        "compile-train-graph": 0, "seed": 7,
        "data-dir": "data", "model-out": "ar.ckpt", "tokenizer-out": "tokenizer.bpe",
    }

    def run(mode="train", expected=0, message="", **changes):
        options = dict(defaults, **changes)
        args = [str(binary)] + (["gen", "int main"] if mode == "gen" else [])
        for key, value in options.items():
            args += ["--" + key, str(value)]
        result = subprocess.run(args, cwd=root, text=True, encoding="utf-8", errors="replace",
                                capture_output=True, timeout=180)
        output = result.stdout + result.stderr
        if result.returncode != expected or message not in output:
            raise RuntimeError(f"Unexpected MLX result ({result.returncode}):\n{output}")
        if expected == 0 and re.search(r"(?<!\w)(nan|[+-]?inf)(?!\w)", output, re.I):
            raise RuntimeError(f"Nonfinite MLX training/generation result:\n{output}")
        return output

    def unpack(path):
        data = path.read_bytes()
        if data[:8] != b"TRTLMX01":
            raise RuntimeError("Missing MLX architecture header")
        length = struct.unpack("<I", data[8:12])[0]
        return json.loads(data[12:12 + length]), data[12 + length:]

    run(message="Final checkpoint saved at completed step 2")
    metadata, payload = unpack(root / "ar.ckpt")
    if struct.unpack("=i", payload[:4])[0] != 2:
        raise RuntimeError("Checkpoint repeated a completed training step")
    run(message="成功恢复状态", steps=3)
    run("gen", message="END", model="ar.ckpt", tokenizer="tokenizer.bpe", **{"max-tokens": 2})
    run("gen", message="END", model="ar.ckpt", tokenizer="tokenizer.bpe", **{"max-tokens": 2, "kv-cache": 0})
    run("gen", 1, "Model checkpoint not found", model="missing.ckpt")
    run(expected=1, message="Invalid model dimensions", dim=15)
    run(expected=1, message="Invalid GaLore configuration", **{"galore-rank": 0})
    run(expected=1, message="MLX checkpoint metadata mismatch: dim", dim=32)
    valid = (root / "ar.ckpt").read_bytes()
    (root / "ar.ckpt").write_bytes(valid[:-1])
    run(expected=1, message="truncated")
    if (root / "ar.ckpt").read_bytes() != valid[:-1]:
        raise RuntimeError("Invalid checkpoint was overwritten")
    (root / "ar.ckpt").write_bytes(valid)
    # Compiled graph and gradient accumulation exercise separate execution paths.
    run(message="Final checkpoint", **{"model-out": "compiled.ckpt", "compile-train-graph": 1, "steps": 1})
    run(message="Final checkpoint", **{"model-out": "micro.ckpt", "packed-batch": 0, "steps": 1})
    # Projected optimizer state, forced refresh and SSD state serialization.
    run(message="Final checkpoint", **{"model-out": "galore.ckpt", "galore-off": 0,
        "galore-rank": 2, "galore-min-size": 64, "galore-refresh": 1,
        "galore-min-interval": 1, "galore-fixed-refresh": 1, "steps": 2})
    run(message="成功恢复状态", **{"model-out": "galore.ckpt", "galore-off": 0,
        "galore-rank": 2, "galore-min-size": 64, "steps": 3})
    run(message="Final checkpoint", **{"model-out": "swap.ckpt", "optimizer-memory": "swap", "steps": 1})
    run(message="成功恢复状态", **{"model-out": "swap.ckpt", "optimizer-memory": "swap", "steps": 2})
    run(message="ELF final checkpoint", **{"model-mode": "elf", "model-out": "elf.ckpt", "steps": 1})
    run(message="ELF EMA restored", **{"model-mode": "elf", "model-out": "elf.ckpt", "steps": 2})
    run("gen", message="END", **{"model-mode": "elf", "model": "elf.ckpt.ema",
        "tokenizer": "tokenizer.bpe", "elf-sample-steps": 2})

    # A small synthetic T5 fixture checks format, shape handling and latent I/O;
    # it is not evidence of quality with an actual T5 encoder.
    vocab, seq, latent = 32, 8, 8
    (root / "latents.bin").write_bytes(
        struct.pack("=7I", 0x5435454C, 1, 1, seq, latent, vocab, 3)
        + struct.pack("=8i", *range(4, 12)) + struct.pack("=64e", *([0.05] * 64)))
    (root / "unembedding.bin").write_bytes(
        struct.pack("=4I", 0x45553554, 1, vocab, latent)
        + struct.pack("=" + "e" * (vocab * latent), *([0.01] * (vocab * latent))))
    t5_options = {"model-mode": "elf", "elf-t5-latents": "latents.bin",
                  "elf-t5-unembedding": "unembedding.bin", "model-out": "t5.ckpt", "steps": 1}
    run(message="ELF final checkpoint", **t5_options)
    run("gen", message="END", **dict(t5_options, model="t5.ckpt.ema", **{"elf-sample-steps": 2}))
    (root / "latents.bin").write_bytes(b"invalid")
    run(expected=1, message="Invalid T5 latent dataset", **t5_options)
print("MLX training/resume/generation/ELF/T5 smoke tests passed")
