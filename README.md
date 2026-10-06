# turtle.AI

Two independent backends are available: the original CPU trainer described below
and the [MLX / OpenMythos backend](mlx/README.md) imported from the local version.
See its [alignment report and next steps](mlx/ALIGNMENT.md) for differences and
validation scope. Their tokenizers and model checkpoints are not interchangeable.

Try to do something.  
  
Hello ! We are turtle . Because of the AI , you need to go to the branch 'in-depth...' to watch our new progress .Thanks for your understanding.  
  
We are new here , please be more understanding about something we not do really well .  
  
(Our English not really well...)

## Build and run

The current checkout contains a C++17 CPU training program and a byte-level BPE
tokenizer. Linux and Windows use OpenMP; macOS uses the Accelerate framework.
The JSON and image-loading libraries are bundled. OpenBLAS and CMake are not
required by this implementation.

Requirements: a C++17 compiler and GNU Make. Tests also require Python 3.
For example, on Debian/Ubuntu install `g++`, `make` and `python3`.

```sh
make -j2
OMP_NUM_THREADS=2 ./build/train config.example.json
make test
make sanitize
```

The example uses the bundled small text corpus and writes models to `build/`.
It demonstrates training; it is too small to produce a useful language model.
For your own corpus, copy `config.example.json` to `config.json` and edit it.
Run commands from the repository root: paths inside the configuration are
relative to the working directory. Create output directories before training.
Running without an argument reads `config.json`.

Configuration constraints:

- `seq_len`, `d_model`, `epochs`, `learning_rate` and `clip_threshold` must be positive.
- `vocab_size` must be at least 260 (four special tokens plus 256 bytes).
  Training can produce fewer tokens when no merge candidates remain.
- The text corpus must contain at least `seq_len + 1` encoded tokens.
- `save_point: 0` disables periodic saves; the final checkpoint is still saved.
- A nonempty `load_path` explicitly requests a checkpoint. Otherwise an existing
  `save_path` is loaded automatically. Use a new path to train from scratch.
- Optional `training.seed` fixes random initialization and sample selection for
  reproducible runs; by default the seed comes from the current time.

Model checkpoints use format version 1: `TRTLMODL`, a little-endian 32-bit JSON
header length, JSON metadata, and model parameters. Metadata records the model
dimensions, block count, float format, byte order, tokenizer fingerprint, and
image-patch layout. Loading rejects mismatches, truncated/oversized payloads,
invalid headers, and nonfinite parameters before changing the model. All linear
weights/biases, embeddings, and both LayerNorm parameter sets are saved. Saving
writes a temporary file and renames it over the destination only after completion.

Legacy raw-weight files are accepted when their byte length matches the current
model. A warning explains that their dimensions and tokenizer cannot be verified;
LayerNorm starts with gamma=1 and beta=0 because the old format omitted it. The
next save upgrades the file to version 1, which older executables cannot read.
Keep a backup of legacy models if you still use those executables. Float payloads
retain native byte order; cross-endian loading is rejected. The separate BPE file
still uses native-size lengths. Resuming restores model parameters; the learning
rate scheduler restarts and randomness comes from the configured seed.

The VLM branch is experimental: it expects a directory with `train.json` (an
array of objects containing `answer`) and images named `train_0.jpg`, etc.
It requires `seq_len > 196` and a nonempty answer string in every sample. Images
are decoded as RGB, resized to 224x224 with bilinear interpolation, and split
into 196 spatial 16x16 patches. Image projection weights and biases receive
gradients from the answer loss. Answers use BOS/EOS tokens so a short answer still
provides supervision. BPE training uses answer text rather than serialized JSON.
Missing or unreadable images fail training with an explicit error. This remains
a small experimental CPU model; training quality on real datasets is unverified.

`make test` checks numerical gradients (including LayerNorm and image projection),
tokenizer reload/retraining and malformed files, image patch order, full parameter
and prediction round trips, legacy checkpoint migration, metadata rejection,
and real text/VLM training with checks that vision and LayerNorm parameters update.
`make sanitize` repeats these checks with AddressSanitizer and UndefinedBehaviorSanitizer.
GitHub Actions also builds the Windows executable.
