# turtle.AI
Try to do something.  
  
Hello ! We are turtle . Because of the AI , you need to go to the branch 'in-depth...' to watch our new progress .Thanks for your understanding.  
  
We are new here , please be more understanding about something we not do really well .  
  
(Our English not really well...)

## Experimental bounded attention (FP32)

The training model accepts two optional keys in `config.json` under `model`:

```json
{
  "attention_window": 128,
  "attention_sink_tokens": 8
}
```

A query attends only to preceding tokens in its local window (including itself)
and the first `attention_sink_tokens` positions of the training sequence. Future
positions are always masked, including future sink positions. The forward and
backward passes use the same mask. Negative settings are rejected.
`attention_window: 0` is the default and preserves full causal attention.
Sink positions reset for each training sample; this is not a persistent KV cache.
All projections, scores, softmax reductions and gradients remain FP32. Existing
checkpoint layouts are unchanged, but keep these configuration values alongside
weights: the attention policy is not serialized into the weight file.

This is an initial model-side step inspired by the endogenous-memory architecture
guide. It still allocates dense N×N score tensors and computes dense products:
**it does not reduce asymptotic compute or memory usage**. It has no learned gist
tokens, reconstruction loss, quantized episodic memory, retrieval, precision gate,
or inference/pause decoder. Local attention changes the available context and
should be trained and evaluated before enabling it for existing weights. No
claim of reduced rank inversion or guaranteed convergence has been validated.

The attention backward pass also now sums the Q, K and V contributions to the
input gradient; previously each projection overwrote the preceding contribution.
The temporary backward tensor is per-call so different sequence lengths are safe.

Build and run the focused numerical checks on Linux with GCC/OpenMP:

```sh
g++ -std=c++17 -O2 -fopenmp train.cpp BPE.cpp -o train.out
g++ -std=c++17 -O1 -fopenmp tests/attention_test.cpp BPE.cpp -o attention_test.out
OMP_NUM_THREADS=1 ./attention_test.out
```

The checks cover causality, sink retention, window boundaries, normalized finite
probabilities, invalid settings, multiple sequence lengths, and input gradients
against finite differences. They do not measure long-context model quality.

## Memory building blocks (not connected to training yet)

`memory.h` provides a dependency-free CPU reference API in `turtle::memory`:

- `StoredVector` stores FP32, symmetric INT8, or packed INT4 vectors with one
  FP32 scale per vector. INT4 uses signed levels -7..7 encoded as offset nibbles;
  this is a custom reference format, not FP8, a standard checkpoint format,
  or a hardware quantized kernel. Decode and inner-product accumulation use FP32.
- `Block` takes caller-defined half-open source boundaries `[begin, end)`, a unique
  ID, retrieval key and payload. Key and payload precision can differ.
- `Store` performs exact maximum-inner-product retrieval, filtering out blocks
  whose end exceeds the query position. Equal scores sort by ID for reproducibility.
  Only selected payloads are decoded; keys are decoded during the linear scan.
  Capacity is an explicit block count; admission fails when full. No eviction or
  automatic precision decisions are imposed. Use a separate store or `clear()`
  for every independent document/session to avoid cross-session retrieval.

`storage_bytes()` reports encoded vector payloads plus quantization scales only,
not container overhead, block metadata, transient buffers, or process memory.
Small vectors may grow because of scale overhead. Search is linear in stored
keys, allocates temporary decoded keys, and is not a scalable ANN index.

```sh
g++ -std=c++17 -O1 -g -fsanitize=address,undefined tests/memory_test.cpp -o memory_test.out
./memory_test.out
```

These components deliberately do not decide chunk boundaries, gist representation,
key learning, precision gating, retention policy, or how retrieved values enter
attention and receive gradients. Those choices await the detailed design record.
Passing component tests establishes encoding/retrieval behavior only, not RAG
quality, rank-inversion reduction, or feasibility of the complete architecture.

## Native Resonant model reference

`models/resonant.py` is a separate, trainable PyTorch implementation of the
native-memory topology. The original C++ trainer and checkpoint format remain
separate; the Python model cannot load its weights. This reference is intended
to validate topology and gradients before a C++/GPU implementation.

Implemented:

- Learned Gist embeddings after completed physical blocks; causal local/block,
  sink and historical-Gist visibility. Source positions are mapped to contiguous
  logical positions. Each query can additionally retrieve Top-K completed blocks
  using its projected Query against their Gist keys.
- Original FP32 K/V for local working memory and anchors. Recalled historical
  body K/V use a learned binary retention gate or symmetric INT4 simulation.
  Selected local and historical keys share one FP32 softmax normalization.
- Straight-through estimators for quantization and retention gates. Hard Top-K
  membership has no gradient; Q/K/V in the selected attention paths do. There is
  no claim of exact end-to-end differentiability through discrete routing.
- Cross entropy, target-vs-competitor margin loss, Gist mean-embedding reconstruction
  proxy, and a retention-cost penalty. The proxy does not guarantee semantic,
  type or symbol preservation. Gate tradeoffs require tuning and evaluation.
- A bounded, batch-one greedy generator with invisible pause inputs, finite pause
  budget per generation request, and explicit low-confidence fallback events.
  Pause embeddings require appropriate training examples; merely enabling the
  mechanism cannot guarantee useful deliberation or increasing margins.
- `PackedINT4` provides actual byte packing for isolated memory storage checks.
  The trainable forward uses fake quantization, not this packed storage class.

Important limits: this teacher-forced training path still uses dense score
matrices and accumulates all historical Gists. The separate streaming path below
implements bounded inference caches and persistent pages. Neither path has GPU
fused kernels, a learned ANN index, automatic SRAM placement, or a constant query
latency guarantee. Recalled low-bit vectors cannot regain lost information through
upcasting. Re-prefill from source can recover unquantized projections under its
new context, but not necessarily the original long-context projections.

```sh
python -m venv .venv
.venv/bin/pip install -r models/requirements.txt --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r tools/requirements-rag.txt
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m tools.experiment_resonant --steps 200 --seeds 3
```

The experiment compares full attention, Gist-only routing and Gist-plus-retrieval
on held-out, random key/value association recall over 20 tokens. It reports three
seeds, recall accuracy, loss, timing and matched-weight cold-FP32 Top-1 disagreement.
This is a short synthetic screening task, not a long-context benchmark, not a
comparison across inference engines, and not proof of rank-inversion suppression.

## External text baseline

`tools/context_memory.py` keeps the alternative text-level JIT-RAG baseline:
C++ Tree-sitter function/declaration chunks, Markdown section/paragraph chunks,
lexical routing, and skeleton/evidence/recent/query assembly. The scheduler
accepts a token-count callable; use the actual target-model tokenizer. Required
base context is never silently truncated; oversized evidence is omitted and
reported. It rebuilds prompts with only current evidence, but makes no model
calls and cannot control inference precision. C++ parsing errors fail explicitly;
macro-heavy code may need preprocessing. It does not resolve C++ imports or
construct a complete dependency graph, nor infer natural-language claims/entities.
Large logical units are preserved, not arbitrarily cut to fit a budget.

`State` requires provenance and an explicit prior value to replace conflicting
facts; `fold_event` preserves a reference to the original observation. These are
caller-controlled primitives, not automatic verified dialogue summarization.
Documents and tool outputs remain untrusted evidence; the consuming agent must
keep system instructions separate from retrieved text.

### Initial screening result (CPU, 3 seeds, 200 training steps each)

| Attention policy | Seed 0 recall | Seed 1 recall | Seed 2 recall | Mean |
|---|---:|---:|---:|---:|
| Full attention | 100% | 100% | 100% | 100% |
| Gist only | 100% | 100% | 73.83% | 91.28% |
| Gist + retrieval | 100% | 100% | 100% | 100% |

Held-out evaluation uses 256 examples per seed, batch 32 during training and a
25% random-guess baseline. Cold-FP32 versus gated quantization Top-1 disagreement
was zero in all nine runs: the test cannot establish inversion reduction because
it produced no inversion events. Full attention also solved this short task.
These results support trainability of the reference, not superiority, long-context
quality, useful pause behavior, or actual GPU memory savings. Separate training
runs also do not isolate causal contributions of every mechanism.


## Bounded streaming KV memory

`models/streaming_memory.py` connects inference-only, batch-one paged memory to
the existing `ResonantLM` weights. `push(token_id)` updates per-layer caches and
returns next-token logits without recomputing the complete prefix. `prefill(ids)`
returns only the latest logits. `generate(...)` uses the same streaming caches,
limits internal pauses, flags low-confidence emissions, and stops explicitly at
positional capacity. Pause usefulness still requires training and evaluation.

```python
from models.streaming_memory import MemoryConfig, ResonantStream

model.eval()  # load trained weights beforehand
memory = MemoryConfig(
    block_size=model.cfg.block_size,
    working_blocks=2, sink_tokens=8, gist_slots=8,
    cold_bytes=65536, top_k=2,
)
stream = ResonantStream(model, memory, page_directory="work/session-pages")
stream.prefill(prompt_token_ids)
result = stream.generate(max_new_tokens=32, max_pauses=3)
print(stream.stats())
stream.reset()  # new session, positions, working/cold pools and disk namespace
```

Policies and bounds:

- Anchors and a bounded number of completed working blocks/Gists remain at
  `working_dtype` (FP32 default, FP16/BF16 supported). An additional pending block
  is bounded by `block_size`. Source vectors are cloned/detached so no sequence
  backing storage or autograd graph is retained. Full blocks get learned Gist
  steps from the network before old blocks move to cold storage.
- Cold K/V are physically packed INT4, with an FP32 scale per token/vector. The
  resident cold pool has a byte budget and FIFO eviction. Cold Gist keys/values
  remain FP32, including in the disk search index. The inference retention policy
  is currently fixed, not the dense training model's learned precision router.
- A Query scores cold Gist keys, loads only selected pages, restores both K/V to
  working precision, then uses FP32 scores, softmax and value accumulation with
  the existing working keys. Duplicate positions are removed. This is exact MIPS
  over Gist keys, not approximate nearest-neighbor search.
- `models/disk_pages.py` writes each demoted page and its index entry to an
  isolated disk namespace when `page_directory` is supplied. Eviction then removes
  only the resident copy. Disk keys are scanned in fixed-size batches, with only
  Top-K candidates retained. No full-history in-RAM catalog is kept. Without disk
  backing, FIFO eviction loses access to old pages: this is a documented baseline.
- `stats()` reports tensor-payload bytes, a conservative persistent-payload bound,
  attention-token bound, resident evictions and disk counters. The bound excludes
  model weights, Python metadata, allocator overhead, temporary packing/gather/
  FP32 attention buffers, source replay cache and disk storage. Each of these
  temporary structures has a bounded block/Top-K/index-batch size, but reported
  payload bytes are **not total process RAM or measured GPU VRAM**. Python object
  counts are bounded except externally persisted history; session counters grow.
- Positions increase monotonically through tokens and Gists. Cached projections
  keep their original position-dependent representation. There is no second
  positional rotation on recall. The learned positional table imposes an explicit
  capacity; overflow is rejected before a token mutates caches. There is no silent
  modulo/reset/position extrapolation. A projection, archive or replay error can
  occur after a partial update; the stream then requires `reset()` before reuse.
- The default legacy dense training policy differs from streaming. For aligned
  training, pass the same `MemoryConfig` to `model.forward`, as described below.
  Exact masks and numerical agreement are tested with evictions, disk backing,
  bounded Gists and FP32/FP16 rounding. No GPU hardware test has been performed.

### Original-token JIT-prefill comparison

`models/source_replay.py` provides `TokenArchive` and `LocalPrefillReplay`:

```python
from models.source_replay import TokenArchive, LocalPrefillReplay

archive = TokenArchive("work/source-blocks", model.cfg.block_size)
replay = LocalPrefillReplay(model, archive)
stream = ResonantStream(model, memory, replay=replay, source_archive=archive,
                         page_directory="work/replay-pages")
```

Complete original-token blocks are written to disk. On retrieval, the callback
replays the selected block through the same model with local positions and a
fresh bounded context. It caches at most one block's per-layer FP32 projections.
This removes quantization from the reconstructed payload, but changes its prior
context and positions, so it is not an exact reconstruction of old K/V. Retrieval
still uses the original cold Gist index. Re-prefill cost and quality require
separate measurement. Resets isolate both source and KV page namespaces. Old
on-disk namespaces are retained for inspection; callers own their disk lifecycle.

### Streaming feasibility screen

```sh
python -m tools.experiment_streaming_memory --seeds 3 --neural-smoke
python -m tools.experiment_streaming_memory --seeds 3 --disk-backed
python -m unittest discover -s tests -p 'test_*.py' -v
```

The controlled paging experiment uses 64-dimensional known random block keys,
32 tokens per block, two working blocks, eight sinks/Gist slots, a 64 KiB resident
cold budget, and Top-2 retrieval. Values are repeated one-hot labels. Every
historical block outside the working window is queried, including evicted pages.
Three seeds, one CPU thread. This deliberately isolates storage/routing; it does
not measure learned semantic retrieval, model reasoning or Top-1 inversion.

| Input tokens | FIFO-only page recall | With disk page recall | Resident KV payload | Disk-backed median query time |
|---|---:|---:|---:|---:|
| 1,024 | 76.67% | 100% | 103.25 KiB | 1.02 ms |
| 4,096 | 18.25% | 100% | 103.25 KiB | 2.05 ms |
| 16,384 | 4.51% | 100% | 103.25 KiB | 3.92 ms |

Disk-backed value accuracy was also 100% in this simple controlled task, across
all seeds/lengths. Disk storage grew from about 164 KiB to 2.78 MiB (serialization
and index overhead included). The full FP32 baseline KV payload grew from 512 KiB
to 8 MiB. Reported times are CPU query timings, averaged per-seed medians; file
system caches are not flushed, so they are not cold-disk or GPU latency claims.

An actual random-weight, one-layer, width-16 model was also streamed over all three
lengths with FP16 working memory and INT4 cold pages: every output was finite,
resident KV payload plateaued at 20.75 KiB (bound 23.125 KiB), and maximum gathered
attention length was 146. This integration run used FIFO-only backing and measures
cache execution, not model accuracy. A separate disk-backed neural integration
run covers 1,024 tokens. Longer histories require a suitably trained positional
table, learned representations and bounded-policy training before quality claims.

Conclusion from this screen: fixed resident storage plus disk history is
implementable, but exact index scanning trades growing query latency and disk
usage for retained historical coverage. Fixed resident memory alone cannot promise
both unlimited recall and constant latency. The next quality experiment should
train/evaluate multi-association retrieval under the same bounded inference policy.


## Aligned-policy multi-association training

`models/memory_policy.py` now simulates the streaming pool during teacher-forced
training: completed working blocks, pending body, protected anchors, limited Gist
slots, exact packed-page byte sizes, FIFO admission/eviction, optional disk history,
and hard Top-K routing with the same tie break. Cold K/V undergo working-dtype
rounding, INT4 simulation and re-rounding before FP32 attention. The policy also
models the Gist's own unrounded projection before storage. This is dense training
with a simulated policy; it does not reduce training memory to the inference bound.

```python
# Native experiment configuration: a Gist summarizes only its own block + sinks.
from models.resonant import Config, ResonantLM
from models.streaming_memory import MemoryConfig, ResonantStream

model = ResonantLM(Config(gist_local_only=True, position_mode="sinusoidal"))
memory = MemoryConfig(block_size=model.cfg.block_size)
result = model(input_ids, answer_targets, memory=memory, disk_backed=True,
               routing_targets=source_block_targets, routing_weight=0.5)
result["losses"]["total"].backward()
# After optimizer steps:
model.eval()
stream = ResonantStream(model, memory, page_directory="work/aligned-pages")
```

`routing_targets` has the same shape as `input_ids`: zero-based source block IDs
at supervised queries, `-100` elsewhere. It is an auxiliary training label, not an
input token and not a production oracle. Later layers learn a differentiable
classification loss over eligible Gist keys; hard Top-K selection remains
nondifferentiable. First-layer Gist keys have not seen content yet and receive no
such classification loss. Bounded training uses the fixed INT4 cold-storage policy,
so its precision-retention gate/loss is inactive. Learned precision allocation is
still a separate task. `disk_backed=True` simulates availability of disk pages;
actual disk I/O is tested by `ResonantStream`, not by the teacher-forced pass.

`gist_local_only=True` makes each summary attend only its own body, anchors and
itself; it does not retrieve other blocks during the summary step. This prevents
summaries from becoming mixtures of unrelated pages. The default remains False
for compatibility with earlier reference checkpoints. Sinusoidal positions are
optional and fixed; they remove the untrained-random-position-table confound when
screening longer inputs, without guaranteeing positional extrapolation quality.

`tools/associative_recall.py` generates new random per-document key/value mappings.
Each eight-token block has a true `PAIR(key,value)` and a contradictory
`IGNORE(key,other_value)` for the same key. The final query asks about one of the
older associations. True keys are unique per document, conflicting values differ,
and no key-to-value mapping is fixed across examples. Query labels and source-page
labels are separate. The CLI can restrict key/value alphabets or remove conflicting
records for calibration; these are different protocols and must not be mixed.

`tools/train_associative_memory.py` trains full, local, limited-Gist, supervised
retrieval or unsupervised-retrieval modes. It saves model, optimizer, RNG state,
configuration and metrics under the specified output directory. Evaluation uses
fresh documents, reports actual answers and page hits separately, compares a
matched-weight cold-FP32 ablation, and checks trained weights through actual
streaming inference. An explicitly labeled oracle-page diagnostic provides the
correct page ID only for an ablation. Production inference does not receive it.

The counterfactual diagnostic keeps the document identical and changes only the
query key to an older fact with a different answer; joint accuracy requires both
answers to be correct. This detects models that merely guess a frequent value
independent of the question. The summarizer adds a query-blind majority baseline
and aggregates all seeds, with ranges rather than selecting the best seed.

Reproduce the fixed three-seed screen:

```sh
python -m tools.train_associative_memory \
  --steps 1000 --seeds 3 --examples 128 --keys 32 --values 8 --layers 3 \
  --local-gists --train-blocks 6 10 14 --eval-blocks 6 10 14 24 32 \
  --stream-blocks 14 32 --modes full gist retrieval --evaluation-seed 3900000 \
  --output work/association-final
python -m tools.train_associative_memory \
  --steps 1000 --seeds 3 --examples 128 --keys 32 --values 8 --layers 3 \
  --local-gists --train-blocks 6 10 14 --eval-blocks 6 10 14 24 32 \
  --stream-blocks 14 32 --modes retrieval_unsupervised --evaluation-seed 3900000 \
  --output work/association-unsupervised
python -m tools.summarize_associations \
  work/association-final/results.json work/association-unsupervised/results.json
```

All modes use the same initial seed, data schedule and language/Gist losses.
Only supervised retrieval receives the extra page-label loss. Therefore a gain
over the full baseline cannot be attributed solely to the memory architecture;
a matched-supervision full-attention control would be needed for that claim.
The unsupervised-retrieval ablation isolates whether the page-label loss is needed
for this particular memory model and training budget. The task is synthetic and
small; no natural-language or code quality claim follows from it.


### Fixed multi-association screen results

Three seeds, 1,000 optimizer steps each, batch 16, width 48, three layers, eight
Tokens per block, 32 possible keys and eight possible values. Training alternates
6/10/14 true associations (50/82/114 input tokens); each has a same-key conflicting
IGNORE record. Queries exclude the two most recent blocks, so answers are outside
the working ring and global Gist slots. Each evaluated length uses 128 fresh
examples per seed. Frozen sinusoidal positions are used in all modes.

The resident policy is one completed working block plus a pending block, one
anchor, two global Gists, four packed cold pages and Top-2 disk-backed retrieval.
Working K/V are FP32 in this quality screen; cold bodies are INT4. This is not a
trained FP16 quality result. Full attention keeps all body/Gist context. Gist-only
and retrieval without page supervision use the same bounded settings as the
supervised mode, with retrieval or page-loss disabled as indicated.

| Input tokens | Full attention | Limited Gist | Retrieval without page loss | Retrieval with page loss | Query-blind majority |
|---|---:|---:|---:|---:|---:|
| 50 | 30.7% | 12.5% | 20.3% | 89.3% | 40.4% |
| 82 | 19.5% | 13.0% | 16.4% | 90.1% | 33.9% |
| 114 | 15.4% | 10.9% | 14.6% | 89.1% | 27.3% |
| 194 | 12.2% | 12.8% | 12.2% | 60.7% | 23.7% |
| 258 | 14.6% | 13.8% | 13.3% | 47.7% | 21.1% |

Values are means across all three seeds, not selected best runs. The chance
baseline over the eight answer values is 12.5%. The supervised mode has substantial
seed variation: at 114 tokens it ranges from 68.0% to 100%, and at 258 tokens from
29.7% to 62.5%. It has not reliably converged after this fixed budget.

Supervised page Top-2 recall averages 99.5% at 114 tokens and 75.3% at 258. The
counterfactual joint-answer accuracy is 79.9% and 42.4%, respectively, supporting
query-dependent lookup rather than a value-frequency shortcut. Providing the
correct page diagnostically raises 258-token answer accuracy only to 49.0%; the
longer-input failure is not just a missing-page problem. This does not isolate
readout, positional extrapolation and representation errors from each other.

Removing page supervision drops both answer accuracy and page selection. The
current prototype therefore needs a source-page training signal: it has not
learned useful retrieval from language loss alone in this budget. The full model
also fails to learn associative behavior well under the current optimization
regime (its majority-guess and counterfactual controls expose that weakness).
These numbers demonstrate a conditional synthetic capability, not a controlled
proof that native memory is superior to a competently trained full Transformer.
A matched-page-supervision full-attention control is still absent.

All 24 selected trained-weight streaming checks match teacher-forced Top-1; the
largest logit difference is about 2.9e-6. These are parity checks, not 24 additional
quality estimates. The supervised model's cold-FP32 ablation has about 0.8% Top-1
disagreement at 114 tokens and 1.8% at 258. Disagreement is not necessarily a harmful
error, and this screen does not establish reduced rank inversion or improved
numerical stability. No adaptive-pause benefit is tested.

Earlier exploratory two-layer, 256-key/32-value runs were near chance for both
full and retrieval models, including an 1,800-step pilot. A smaller-alphabet
calibration and block-local Gists preceded the fixed final three-layer protocol.
Final evaluation uses a separate seed set. Those pilots motivate the controlled
screen but do not establish why the later model learns: several settings changed.

The next quality step is longer-context training with source-page labels (or a
validated self-supervised way to derive them), plus readout-focused diagnostics
and a full-attention baseline with matched auxiliary supervision. The 258-token
quality screen is distinct from the earlier 16K random-weight cache execution
and known-key paging tests; no learned 16K or real-document capability is claimed.

Load a saved experimental model together with its exact memory policy:

```python
from models.checkpoints import load_model_and_memory
from models.streaming_memory import ResonantStream

model, memory = load_model_and_memory("work/association-final/retrieval-seed0/checkpoint.pt")
stream = ResonantStream(model, memory, page_directory="work/loaded-pages")
# Token IDs must follow the checkpoint's synthetic vocabulary/task format.
```

## Source-derived labels and matched supervision

`tools/source_labels.py` extracts explicit PAIR facts, ignores conflicting IGNORE
records, and preserves half-open token spans plus page IDs. `generate_examples`
constructs a bounded number of questions/answers from such a document;
`derive_batch_targets` independently reconstructs labels for existing questions.
The training CLI now uses those derived labels and verifies them against the
synthetic generator as a consistency check. Malformed records, duplicate true
keys, missing query facts, and facts crossing pages fail explicitly. This is a
controlled record grammar, not a natural-language fact extractor or text tokenizer.

`full_supervised` adds exactly the retrieval model's page classification objective
(weight 0.5, same eligible pages and source labels) to full attention. The new
`supervision_policy` controls loss candidates separately from the actual attention
policy: it never restricts the full model's context or inserts oracle pages. Unit
tests verify unchanged logits/visibility with and without auxiliary labels, and
nonzero auxiliary gradients. Inference checkpoints retain the full memory policy.

A longer-distance preliminary comparison trains on 6/14/32 records and evaluates
14/32/64/128 records (114/258/514/1026 input tokens), with 128 possible keys and
eight values. It uses one seed, 600 steps, batch eight, three width-48 layers and
32 held-out examples per length. It changes vocabulary size, lengths and training
budget relative to the earlier three-seed screen, so its results cannot be compared
as a controlled improvement over that screen.

```sh
python -m tools.train_associative_memory \
  --steps 600 --seeds 1 --batch-size 8 --examples 32 --keys 128 --values 8 \
  --layers 3 --local-gists --train-blocks 6 14 32 --eval-blocks 14 32 64 128 \
  --stream-blocks 14 128 --modes full_supervised retrieval \
  --evaluation-seed 4900000 --output work/source-supervised-long
```

The one-seed longer-distance screen completed with 21 regression tests passing.
Both equally supervised models remained near the 12.5% value-only chance level:

| Input tokens | Full + page loss | Bounded retrieval + page loss |
|---|---:|---:|
| 114 | 12.5% | 12.5% |
| 258 | 9.4% | 9.4% |
| 514 | 9.4% | 9.4% |
| 1026 | 15.6% | 15.6% |

This larger-alphabet, longer-curriculum run did not establish learned associative
recall within 600 steps. It cannot decide which architecture is better, and does
not overturn the earlier smaller-alphabet result. The full-supervision control is
now implemented, but still needs a competent optimization regime and multiple
seeds before a meaningful architecture comparison. Source labels are mechanically
verified; their correctness alone does not solve representation learning. A useful
next experiment holds vocabulary and supervision fixed, establishes competent
short-context training for both models, then gradually increases distances.

## Validation-gated fixed-vocabulary curriculum

`tools/curriculum_memory.py` fixes 128 possible keys, eight values, conflicting
IGNORE records, architecture, auxiliary supervision and optimizer. Both full
attention and bounded retrieval receive the same source-derived page labels and
loss weight. Training starts at six records, then attempts 14 and 32. Positional
capacity is allocated once before training; neither vocabulary nor model shape
changes between stages. Every fourth step rehearses the preceding length.

Promotion requires at least 95% accuracy on two consecutive fresh 64-example
validation sets, checked every 300 steps. Each stage has a hard 1,800-step budget;
failure stops escalation. This empirical gate is not a statistical confidence
bound. Validation seeds, training RNG and the final 128-example-per-length test
set are separate. Final tests do not select stages or checkpoints. The checkpoint
preserves the optimizer, RNG, configuration and stage history.

```sh
python -m tools.curriculum_memory --output work/curriculum-fixed-vocabulary
```

The default runs three seeds for both architectures. Adaptively stopping at
competence creates different training budgets; use reported steps and stage
histories, not just final accuracy, when interpreting comparisons. The experiment
checks whether a training curriculum becomes workable, not a compute-matched
architecture ranking. A model failing the short-stage gate is not sent into more
expensive long training merely to produce a long-context score.

`tools/diagnose_single_page.py` is a separate readout diagnostic: one PAIR and a
same-key contradictory IGNORE record, followed by a question. There is no choice
of historical page. It trains one three-layer model for 800 steps with the same
128-key/eight-value vocabulary, then evaluates 256 fresh examples. This is not a
memory benchmark or an architecture comparison; it tests whether the record-role
and value-output mechanics can be learned before addressing multi-page binding.

```sh
python -m tools.diagnose_single_page
```

The curriculum uses microbatches capped by `1024 // input_length` (at most 16),
so its 6/14/32-record stages use batches 16/8/3. This is the same rule for both
architectures, but changes optimization noise as distance grows; a failed longer
stage cannot be attributed to context length alone. A constant-effective-batch
follow-up would require accumulation and a separate compute budget.

The single-page diagnostic also supports `--shuffle-roles`, independently
randomizing whether PAIR or IGNORE comes first. Both the fixed-order and shuffled
one-seed runs achieved 100% on 256 held-out examples after 800 steps. A strategy
that simply chooses one of the two observed values has 50% chance, compared with
12.5% for a uniform guess over all eight value classes. The shuffled result rules
out a fixed-slot copy shortcut in this diagnostic; it does not establish
multi-page key binding or natural-language role understanding.

`tools/evaluate_curriculum_extension.py` can extend the frozen sinusoidal table
of a checkpoint that passed every curriculum gate, without changing learned
weights. It refuses learned-position checkpoints and verifies the preserved
sinusoidal prefix. Full attention is given enough cache capacity to stay full at
the longer length. This is explicitly an untrained-distance diagnostic; it does
not turn a short-trained model into a validated long-context model.

```sh
python -m tools.evaluate_curriculum_extension \
  work/curriculum-fixed-vocabulary/full_supervised-1/checkpoint.pt
```

The qualified full-attention seed-1 checkpoint scored 43.75% at 514 tokens and
18.75% at 1,026 tokens (32 new examples each). It was selected by curriculum
validation gates, not by extension scores. There is only one such checkpoint in
this screen, and these small-sample extrapolation results cannot establish an
architecture ranking or a numerical rank-inversion mechanism.

### Curriculum screen outcome

All six runs completed. With the fixed 128-key/eight-value vocabulary and shared
page supervision, full attention passed the first gate in 1/3 seeds and all three
gates in 1/3; bounded retrieval passed the first gate in 0/3. Runs that failed
remained at six records and stopped after 1,800 steps. Full seed 1 used 4,200 total
steps and achieved 100%/99.22%/96.88% on independent 50/114/258-token tests. The
other full seeds also failed the first gate, so this is not reliable training.

The selected success is evidence that a full baseline can learn under a suitable
trajectory, correcting the earlier impression from short-budget comparisons. It
is not a compute-matched superiority claim: the successful run consumed additional
stages and training. The retrieval model still needs a stable cross-page binding
curriculum before long-range benefits can be assessed. Single-page readout, even
with randomized record order, is learnable; success on that diagnostic does not
remove multi-page representation/optimization difficulties.

The next bounded experiment should isolate two- or three-candidate-page binding,
with explicit source labels and constant effective batch size, then increase
candidate count only after stable validation. Do not interpret the present failures
as proof of quantization harm or an architectural impossibility.

## Two-/three-candidate binding screen with fixed effective batch

`curriculum_memory.py --effective-batch-size 16 --microbatch 4` accumulates four
microbatch gradients before one optimizer update. Remainders are weighted by
actual example count. A numerical test compares full-batch gradients with an
uneven 2+2+1 split, including language, Gist and page-routing losses.

```sh
python -m tools.curriculum_memory --stages 4 5 6 10 \
  --stage-steps 1200 --check-every 200 --effective-batch-size 16 --microbatch 4 \
  --validation-seed 1720000 --test-seed 10300000 \
  --output work/candidate-binding-curriculum
```

Vocabulary, source supervision, decoys and model stay fixed. Four/five/six/ten
source blocks give two/three/four/eight possible answer-source blocks because the
two most recent blocks cannot be queried. There may be more eligible cold pages
than answer-source candidates. Top-2 can cover both possible answer pages in the
first stage without meaningful key-based page ranking, so page recall there is
not evidence of learned retrieval. Answer accuracy and changed-query joint
accuracy diagnose binding. Further stages and the existing two-pass 95% gate
prevent interpreting a two-candidate success as general memory competence.

The old variable-batch behavior remains the default for reproducing earlier
runs. This new protocol uses fresh validation/test seeds and is a different
experiment; its improvement, if any, cannot isolate a single changed setting.

`tools/diagnose_page_competition.py ROOT` evaluates an offline known-page-only
mask: annotated query rows can read only their true source page and themselves.
It grants access even if retrieval missed the page and removes competing pages.
This is an intervention using ground truth, not a production feature or guaranteed
upper bound. The default model behavior is unchanged; a test verifies exact
visibility and causality in the diagnostic.

`tools/continue_binding.py ROOT --output PATH` is a separately reported follow-up:
it selects failed checkpoints whose last two validation scores are at least 75%
and nondecreasing, restores model/optimizer/RNG, and trains the current length for
at most 1,200 additional steps with effective batch 16. Fresh validation and test
seeds are used. It does not retroactively change the fixed-budget pass rate and
must not be interpreted as an unbiased estimate across all seeds.

### Fixed-effective-batch screen and continuation

At the initial 1,200-step-per-stage budget, neither architecture passed the
first two-answer-candidate gate in any of three seeds. The best retrieval seed's
validation was still rising (82.8%, then 90.6%), so this was a budget-limited
failure, not evidence that it cannot learn binding.

The predefined validation-only continuation rule selected retrieval seed 1.
Restoring its model/optimizer/RNG and adding 600 steps yielded validation scores
93.75%, 96.875%, 98.4375%; it passed the two-consecutive-95% rule. A fresh
128-example test scored 98.4375%, with 95.61% joint correctness on changed-query
pairs. This selected-model continuation is separate from the original 0/3 pass
rate and does not establish seed robustness or architectural superiority.

On separate 64-example diagnostic sets, granting only the known page increased
weak full-model readout from 53.1–56.3% to 78.1–89.1%, and weak retrieval seeds
from 50.0–53.1% to 89.1–90.6%. The stronger retrieval seed went from 90.6% to
93.75%. Removing page competition helps, but these oracle interventions do not
identify a single causal fault or represent achievable retrieval accuracy.

After a continuation passes, `--advance` increases the number of source blocks
by one while restoring the exact optimizer/RNG and retaining effective batch 16.
It refuses advancement beyond the checkpoint's positional capacity. Fresh
validation and test seeds are used; this is another explicitly selected-model
follow-up, not a new independent-seed screen.

```sh
python -m tools.continue_binding work/candidate-binding-curriculum \
  --output work/binding-continuation
python -m tools.continue_binding work/binding-continuation --advance \
  --output work/three-candidate-advance
```

Advancing the selected retrieval seed from two to three answer candidates used
1,200 further steps without old-length rehearsal. It did not pass the gate:
validation ended at 82.81%, and the independent 128-example test scored 68.75%,
with 98.44% page recall and 46.46% changed-query joint accuracy. The old two-source
retention test after this adaptation scored 78.125%. This exposes both imperfect
readout despite high page coverage and a retention concern; the continuation is
not a reliable general binding solution. Next isolate old-length rehearsal and
readout competition before increasing candidate count further.

An oracle-inclusion diagnostic bug was corrected: when a target was already
second in Top-K, replacing the first slot with it duplicated the target and
silently removed a distractor. It now leaves an already-present target untouched.
A regression test verifies identical masks/logits in that case. Historical
`oracle_page_accuracy` numbers before this correction mixed correct-page access
with reduced competition and must not be treated as pure inclusion evidence.
Normal answer scores, page recall and training were unaffected. The corrected
three-candidate oracle-inclusion result is 71.875%, versus the old mixed result
85.9375%; it no longer suggests that simple page access explains the main deficit.

All 25 tests pass, including gradient accumulation and oracle-mask checks.

On the exact same 128-example retention set, the two-candidate checkpoint scored
99.21875% before three-candidate adaptation and 78.125% after it. This paired
comparison supports forgetting under the no-rehearsal advance protocol, rather
than attributing the drop only to different test samples.

## Paired old-task replay adaptation

`tools/replay_binding.py` branches from the same validation-qualified two-candidate
checkpoint, including its optimizer state. Three adaptation-data seeds are used;
these are not independent pretrained initializations. Every branch runs a fixed
800 optimizer updates, with four microbatches of four and no best-checkpoint
selection. New and old training RNGs are independent, and final tests are fresh,
paired before/after and shared across branches.

- `new_only`: 16 three-candidate examples per update.
- `replay`: eight identical new examples plus eight old two-candidate examples.
- `half_new`: the same eight new examples with half-normalized gradient, plus
  zero-weight old forwards/backwards. This controls reduced new-task exposure and
  gradient scale; its effective nonzero-loss batch is eight, not sixteen.

The replay/half-new contrast isolates adding the old-task loss under matched new
exposure. The new-only contrast measures the practical full new-data adaptation.
All use the same number of optimizer updates and sample slots, but differing
sequence lengths mean the FLOP budgets are not exactly equal. Validation logs do
not stop training. Answer accuracy, changed-query joint accuracy, page recall and
known-page-only diagnostics are reported separately.

```sh
python -m tools.replay_binding work/binding-continuation/retrieval-1.pt \
  --output work/paired-replay-binding
```

### Paired replay outcome

All nine adaptations completed. These are three adaptation-data seeds from one
selected pretrained initialization, with 800 fixed updates per branch. Fresh
paired tests use 128 examples per task/seed. Before adaptation, mean accuracy was
97.14% on the old two-candidate task and 62.24% on the new three-candidate task.

| Adaptation | Old task mean | New task mean | Old changed-query joint | New changed-query joint |
|---|---:|---:|---:|---:|
| New only | 77.34% | 64.58% | 52.17% | 47.90% |
| Half-new gradient control | 79.43% | 63.80% | 53.91% | 48.38% |
| Old/new replay | 97.66% | 73.96% | 95.15% | 66.32% |

Replay improved both task accuracies relative to new-only in all three adaptation
seeds. Its ranges were 94.53–99.22% old and 70.31–78.13% new. Reduced new exposure
alone did not reproduce the benefit. This supports using old-task loss for this
adaptation regime, but is not an independent-initialization robustness result,
proof of universal replay benefit, or a long-context capability result.

`tools/analyze_replay.py ROOT` verifies that branch before-test accuracies match,
aggregates all seeds, and recomputes paired normal/page-only diagnostics on the
exact same separate inputs. For replay models, mean normal/page-only scores were
98.96%/83.07% on the old task and 70.05%/81.77% on the new task. Thus removing
competing context can help the harder task but harm a learned task. This
intervention also changes the query's context distribution and cannot isolate a
unique mechanism. It motivates selective readout retaining useful global context,
not a blanket policy of reading only one page.

All 26 regression tests pass. Checkpoints record adaptation provenance separately
from the parent checkpoint's qualification; old pass flags are not reused to
claim that the new task passed. The new task still does not meet the 95% gate.

## Residual selective readout fusion

`Config(readout_fusion=True)` adds one per-query sigmoid gate per attention layer.
For ordinary queries, let `J` be the original joint attention read (working memory,
anchors, Gists and selected history) and `R` the independent softmax read over
visible recalled positions outside the working set. The result is
`(1-g) J + g R`, where `g = 0.5 * sigmoid(Wx+b)`. Initial `g` is 0.05. Thus at least
half the original joint read remains at each layer. This is a probability/readout
constraint, not a guarantee of global semantic preservation across the network.

Empty recall sets use exactly `J` with no NaN or gate effect. Gist summary queries
remain unchanged. The feature requires an explicit memory policy; the legacy
implicit dense path rejects it. Real streaming caches classify recalled positions
using the same working-set membership and apply the same gate, with FP32 score
and value accumulation. The existing packed INT4 storage, recall budget and causal
visibility are unchanged. Learned gate parameters are additional model weights;
there is no fused GPU kernel or memory-saving claim for this experiment.

`tools/train_readout_fusion.py` migrates an old checkpoint explicitly, permitting
only the new gate weights to be missing. Existing optimizer moments are restored
for original parameters; learned gates receive their own new optimizer group.
Three adaptation-data seeds compare joint attention, a fixed 0.05 residual mix,
and learned gating. All use identical 8-old/8-new replay, microbatch four, 800
fixed optimizer updates, source-page supervision and fresh paired tests. Fixed
and learned variants start with the same function. These seeds share a single
pretrained initialization; they do not establish initialization robustness.

```sh
python -m tools.train_readout_fusion work/binding-continuation/retrieval-1.pt \
  --output work/readout-fusion-screen
```

New tests cover empty-recall behavior, probability normalization, preserved joint
mass, nonzero gate gradients and actual streaming/teacher-forced agreement.

### Fusion screen result

All nine fixed-budget adaptations completed. Means across the three paired
adaptation-data seeds (one pretrained initialization):

| Readout | Old task answer accuracy | New task answer accuracy | New changed-query joint |
|---|---:|---:|---:|
| Original joint | 96.09% | 70.31% | 56.85% |
| Fixed 0.05 mix | 96.61% | 71.35% | 54.72% |
| Learned mix | 95.31% | 71.09% | 53.63% |

The learned/new-task paired differences from joint were -0.78, +2.34 and +0.78
percentage points: three additional correct answers out of the combined 384,
with three fewer old-task answers correct. No significance or consistent gain
is established. The learned variant also did not improve changed-query joint
accuracy. These are new paired adaptation/test seeds; comparing this 70.31% joint
baseline directly with the earlier replay screen's 73.96% would confound data
variation with architecture changes.

The gates were active, not collapsed: mean recalled-read fractions across the
three layers were about 0.255/0.340/0.181 for old-task queries and
0.433/0.412/0.270 for new-task queries. This shows task-dependent weighting in
this synthetic distribution, not semantic understanding or proof that the
weighting solves binding. Initial fixed and learned migration functions were
verified to give the same before-test answer scores. Zero-shot new-task means
were 60.42% joint and 60.68% with either initialized fusion variant.

All 18 selected trained streaming checks matched teacher-forced Top-1; the maximum
logit difference was about 1.15e-5. The small parity tests additionally cover FP16
working storage, causal Gist behavior, empty retrieval and gate gradients. These
are parity checks rather than evidence of numerical rank-inversion suppression.

The fusion flag remains **False by default**. This bounded residual variant is a
reproducible negative screen, not a recommended replacement for joint attention.
Recall still combines up to two pages inside one recalled-read softmax; adding a
working-vs-recalled gate does not itself resolve key/value binding or competition
inside recalled pages. A page-score-conditioned reader is a separate hypothesis
that still requires a controlled experiment.

```sh
python -m tools.analyze_readout_fusion work/readout-fusion-screen
```

## Page-conditioned, recall-mass-preserving readout

`Config(page_score_weight=1.0)` scores the already selected pages using the query
and their high-precision Gist keys, then adds their relative scores to recalled
key logits. The selected page scores are centered; with one selected page the
bias is zero. Protected working tokens and global Gists are not directly biased.
No new model parameters or page budget are introduced. The default weight is zero.

The implementation adds a recalled-region log-normalizer correction. For a fixed
query/KV set, the recalled exp-sum is kept equal to the original, so working-region
probabilities and total recalled probability remain unchanged (up to floating-point
rounding). Only distribution *inside* visible recalled keys changes. This does not
guarantee semantic retention across layers or prevent representation changes in
training. A preliminary raw-bias run was stopped because it confounded page
selection with changing total recalled mass; it is not the reported experiment.

`page_score_detach=True` stops gradients through the added page scores, while
normal token attention and the existing page-classification objective still train.
With False, answer loss also backpropagates through selected-page scores. Hard
Top-K membership remains nondifferentiable. Both variants have the same initial
forward function, allowing comparison of the new answer-to-page-score gradient.

Dense and actual streaming paths use the same selected Gist keys, centering and
normalizer compensation. No future page can enter the causal visibility mask;
empty recall has an exact safe fallback. The storage format stays packed INT4
for cold bodies and the working/Gist policy is unchanged. Tests verify local
probability conservation, empty-region finite gradients, single-page invariance,
causal prefixes and streaming parity.

```sh
python -m tools.train_page_readout work/binding-continuation/retrieval-1.pt \
  --output work/page-readout-mass-neutral
python -m tools.analyze_page_readout work/page-readout-mass-neutral
```

The three variants are joint attention, detached score conditioning, and
differentiable score conditioning. Each starts from the same qualified ungated
checkpoint/optimizer, uses identical old/new replay, and trains for 800 fixed
updates under three adaptation-data seeds. They share one pretrained initialization;
this does not establish initialization robustness. Fresh validation/test seeds
are used, and final weights are tested without best-checkpoint selection.

### Page-conditioned readout result

All nine mass-neutral runs completed. Means across three paired adaptation-data
seeds from the same qualified initialization:

| Readout | Old task accuracy | New task accuracy | New changed-query joint | New page recall |
|---|---:|---:|---:|---:|
| Original joint | 94.79% | 69.27% | 60.88% | 98.70% |
| Forward conditioning, detached scores | 94.27% | 69.79% | 60.16% | 98.70% |
| Differentiable conditioning | 95.31% | 70.83% | 60.35% | 97.92% |

Differentiable/new-task paired gains were +2.34, +4.69 and -2.34 percentage points,
six extra correct answers among the combined 384. The direction was not consistent
across data seeds, and changed-query joint accuracy did not improve. This is a
small exploratory screen, not evidence of a reliable architecture gain or a
significance claim. It uses new paired data/test seeds, so comparisons with earlier
replay/fusion tables do not isolate a mechanism change.

All 18 selected trained streaming checks matched Top-1; the maximum logit difference
was about 3.82e-6. Unit checks additionally cover FP16 working storage and verify
that detached/differentiable conditioning has identical forward values but
correctly different gradients. The mass compensation preserves working-region
probabilities both forward and with respect to the added coarse score bias.
All 33 regression tests pass. This is not evidence of rank-inversion suppression.

The default `page_score_weight` remains zero. The tested operation reweights
*between* recalled pages while leaving each page's conditional token ordering
unchanged. It does not directly teach which same-key PAIR/IGNORE value is correct.
High page coverage alongside weak answers therefore motivates checking whether
Gist representations preserve the correct key/value relation and whether the
reader can recover it. The mean-embedding Gist reconstruction proxy has never
guaranteed relation preservation. A representation/readout probe should precede
more claims about attention weighting or long-context quality.

The stopped raw-bias pilot is marked ABORTED in its work directory. Reported
numbers are only from `page-readout-mass-neutral`; checkpoints record the
`mass-neutral-v1` protocol separately from parent training provenance.

## Frozen memory relation probes

`tools/probe_gist_relations.py` freezes every encoder weight and trains only small
linear/MLP readout probes. Training/validation/test documents use independent RNGs;
normalization uses training features only; validation loss selects a probe from a
fixed 600-step budget. Three probe initializations share the same data and frozen
encoder, so they do not measure encoder-seed robustness. Encoder parameters are
cloned and checked unchanged after probing.

Representations include final normalized Gist hidden states, contextual body
means, sorted raw embedding bags, and an explicitly oracle-selected true-value
embedding. The oracle feature is a positive optimization control, not an achievable
reader feature. Sorted bags deliberately discard role relationships and avoid
floating-reduction-order artifacts. Candidate values are eight classes, keys 128;
absolute key recovery and value recovery are distinct from native query-relative
retrieval. A failed global key classifier does not prove that dot-product routing
cannot identify a page within its current document.

Crucially, final Gist outputs are not the actual per-layer stored KV projections.
The `gist_k_N`, `gist_v_N`, `gist_kv_N` features capture each layer's QKV projection
at Gist positions, cast to the configured working dtype. Tests compare extracted
features directly with actual streaming Gist keys/values. Layer-zero Gist K is a
constant-position negative control, because it has not attended to block contents.

Probe documents either randomize PAIR/IGNORE record order or keep the model's
original fixed format. Paired whole-record swaps preserve semantic labels; fixed
and longer documents are additional tests. These controlled order shifts are not
natural-language evaluations. Label-shuffle, random-encoder and oracle-value
controls are reported. Weak decoding cannot prove information-theoretic loss:
probe capacity, optimization, encoder distribution shift and query conditioning
remain possible confounds.

```sh
python -m tools.probe_gist_relations \
  --parent work/binding-continuation/retrieval-1.pt \
  --replay work/paired-replay-binding/replay-0/checkpoint.pt \
  --output work/gist-relation-probes
python -m tools.probe_gist_relations \
  --parent work/binding-continuation/retrieval-1.pt \
  --replay work/paired-replay-binding/replay-0/checkpoint.pt \
  --features gist_k_0 gist_k_1 gist_v_1 gist_kv_1 gist_k_2 gist_v_2 gist_kv_2 \
  --output work/gist-kv-probes
python -m tools.probe_gist_relations \
  --parent work/binding-continuation/retrieval-1.pt \
  --replay work/paired-replay-binding/replay-0/checkpoint.pt \
  --features gist gist_kv_1 gist_kv_2 --train-roles fixed \
  --output work/gist-fixed-role-probes
```

`tools/probe_candidate_relations.py` removes key/vocabulary classification as a
competing objective. It gives a bilinear or MLP probe the frozen representation
plus a candidate's frozen value embedding and asks whether the candidate belongs
to PAIR or IGNORE. Positive/negative candidates are paired and balanced. Their
identities come from source annotations: this is a diagnostic, not a deployable
reader. Probe training/test documents and encoder weights remain separate/frozen.

```sh
python -m tools.probe_candidate_relations \
  --parent work/binding-continuation/retrieval-1.pt \
  --replay work/paired-replay-binding/replay-0/checkpoint.pt \
  --output work/candidate-relation-probes
# Repeat with --train-roles fixed and a distinct output path.
```

A marker-only counterfactual unit test swaps which value is true while keeping the
sorted token bag exactly identical. Any deterministic bag+candidate scorer must
average 50% pair choice over this paired case. This tests the role-information
negative control independently of how well a particular probe optimizes.

### Actual-memory and whole-page probe outcomes

The first final-hidden-state probes were weak, but that vector is not what native
attention stores. Direct per-layer K/V probing and stream-cache equality checks
avoid treating terminal Gist outputs as the whole memory. First-layer Gist keys
were near chance as expected. Global key-ID classification also need not match
the native query-relative MIPS task; low key classification is not proof that a
page cannot be located within its document.

For the frozen replay seed-0 encoder, last-layer Gist K/V candidate-relation MLPs
scored 77.7% under fixed-order probe training/testing, but only 56.6% when trained
and tested with randomized record order. These averages use all source blocks.
Probe training accuracy was also limited (79.2% fixed, 61.3% mixed), so probe
optimization/capacity and encoder distribution shift remain confounds. This does
not establish that the information is irreversibly lost.

A matched follow-up restricts probes to the first three historical answer-source
pages, outside the two most recent pages. It includes the entire ordered per-token
K/V array and an INT4-encoded/dequantized version. Protected anchors remain at
working precision. Tests compare the reconstructed page vectors with actual cold
pages and their Gist index K/V. Raw and INT4 features share the same frozen encoder
path: this isolates the page storage conversion, not every effect of low-precision
attention during encoder execution.

```sh
python -m tools.probe_candidate_relations \
  --parent work/binding-continuation/retrieval-1.pt \
  --replay work/paired-replay-binding/replay-0/checkpoint.pt \
  --features gist_kv_2 body_kv_2 body_int4_kv_2 raw_bag oracle_value \
  --historical-only --output work/historical-body-relation-probes
```

On the same historical-page population, MLP candidate-pair choice averaged 67.9%
from Gist K/V, 95.6% from ordered whole-page K/V, and 93.9% after INT4 storage.
However, a candidate probe is given the two actual values by source annotations.
Ranking all eight values with that binary-trained head gave only 26.2%, 59.4%,
and 58.5%, respectively. Thus candidate-pair success is not a native answer score,
and binary candidate training does not adequately constrain unseen alternatives.

`tools/probe_direct_value.py` removes that objective mismatch: it trains an
additional eight-class value reader directly on frozen page representations,
with no candidate identities supplied. Same independent document seeds and
historical-page restriction; 600 probe steps, three probe initializations,
validation-only selection, 512 held-out documents/1,536 historical pages.

```sh
python -m tools.probe_direct_value \
  work/paired-replay-binding/replay-0/checkpoint.pt \
  --output work/direct-page-value-probes
```

| Frozen representation | Eight-class accuracy, randomized roles | Fixed-order test |
|---|---:|---:|
| Last-layer Gist K/V | 50.35% | 71.68% |
| Ordered whole-page K/V | 94.79% | 97.37% |
| Ordered INT4 page K/V | 92.36% | 96.27% |
| Sorted embedding bag | 51.56% | 49.07% |
| Oracle true-value embedding | 100% | 100% |

The uniform eight-class chance is 12.5%. The bag contains both observed values,
so 50% candidate ambiguity is the relevant stronger negative control. The
oracle is label-selected and merely checks optimization. Parent, replay and
random-encoder controls and label permutation results are preserved in the work
JSONs. All frozen encoder weights were verified unchanged.

These are extra readers, trained with dense page-level labels and known correct
page boundaries. Whole-page readers also have a larger input/parameter count than
Gist readers. Their success proves recoverability in this synthetic setting, not
that the native model already reads at 92.36%, that a capacity-matched reader will
succeed, or that arbitrary long-text quantization is harmless. Three probe seeds
share one frozen checkpoint and test set; they are not independent model trials.

The useful engineering conclusion is narrower: the tested INT4 pages still carry
readable relation information. The native reader/training path has not exploited
it reliably. The mean-embedding reconstruction proxy is invariant to semantic
marker flips, so it supplies no direct PAIR-versus-IGNORE relation constraint.
A controlled next step is explicit per-page relation reading/reconstruction
supervision with a matched-supervision baseline, rather than assuming that a
failed answer demonstrates destroyed cold-memory information.

Probe heads from the direct-value and all-value diagnostic runs are saved with
feature/candidate normalization metadata; they are diagnostic heads, not native
LM checkpoints. All 36 regression tests pass, including source-semantic controls,
historical-page selection and actual-cache feature checks.

## Native per-page relation reconstruction supervision

`Config(relation_reader=True)` adds an auxiliary page reader over the final
attention layer's ordered K/V projections. The feature uses the same simulated
INT4 cold storage and protected-anchor precision for both bounded and full models.
`relation_targets` contains one source-derived value-token ID per completed page;
`relation_weight` adds its cross entropy to training. The main LM logits do not
consume labels or the auxiliary head's predictions. At inference without relation
labels, the page-reader computation is skipped; this is a training-objective
experiment, not an oracle answer replacement.

`tools/train_relation_auxiliary.py` compares full attention and bounded retrieval,
each with relation loss 0 or 0.1, under the same page classification supervision,
initial backbone/optimizer, replay batches and adaptation-data seeds. All four
conditions instantiate the same auxiliary head; it has zero loss weight in the
control. Head parameters have a new optimizer group; original optimizer moments
are preserved. The extra loss may affect clipping/optimization, so an improvement
would not by itself isolate a single mechanism.

```sh
python -m tools.train_relation_auxiliary \
  work/binding-continuation/retrieval-1.pt \
  --output work/relation-auxiliary-screen
```

The default uses three adaptation-data seeds, 600 fixed updates, effective batch
16 (8 old + 8 new), and fresh validation/test seeds. Both architectures start from
a checkpoint originally trained with bounded memory; this is a paired adaptation
comparison, not training two architectures from scratch. No best-checkpoint
selection is used. Page labels come only from explicit source records. A regression
test verifies that auxiliary labels leave answer forward logits unchanged and
that the relation loss reaches both the reader and backbone KV projections.

### Relation-supervision screening results

The fixed-order screen used three adaptation-data seeds (one shared pretrained
backbone). Mean native answer accuracy on five-page documents was 66.15% without
the auxiliary loss versus 64.84% with it for full attention, and 66.41% versus
65.62% for bounded retrieval. The trained auxiliary heads scored 100% on the
known correct page in the original layout, but only 0--1.30% after swapping the
whole PAIR and IGNORE records within every page. Native answers also collapsed
on that swap. Fixed-layout success therefore included a positional shortcut;
this screen does not veto relation supervision or establish semantic reading.

The smallest follow-up randomized whole-record order independently per training
page in **all four conditions**, keeping source-derived answers and page labels
unchanged. It used one adaptation seed, the same 600 updates, and a new held-out
test seed family. Fixed, mixed (independent per-page swaps), and all-swapped test
views share the same source documents within this pilot.

```sh
python -m tools.train_relation_auxiliary \
  work/binding-continuation/retrieval-1.pt --shuffle-roles --seeds 1 \
  --test-base 47300000 --validation-base 46300000 \
  --output work/relation-auxiliary-order-pilot
python -m tools.analyze_relation_auxiliary work/relation-auxiliary-order-pilot
```

| Random-order training, five-page test | Fixed native | Mixed native | Swapped native | Mixed auxiliary head on true page |
|---|---:|---:|---:|---:|
| Full, loss 0 | 64.84% | 67.19% | 65.62% | Untrained |
| Full, loss 0.1 | 66.41% | 63.28% | 55.47% | 97.66% |
| Retrieval, loss 0 | 65.62% | 67.97% | 66.41% | Untrained |
| Retrieval, loss 0.1 | 65.62% | 66.41% | 66.41% | 91.41% |

The randomized auxiliary readers can recover the relation across record order,
but the native LM has no demonstrated benefit from this auxiliary loss. The full
model also regressed on the four-page retention task (95.31% to 85.16% in the
fixed test view). Randomized training alone removed the catastrophic order-swap
failure in the zero-loss controls. The pilot changes the training distribution
and is only one seed; comparisons against the preceding fixed-order screen do
not isolate the effect of the auxiliary loss. Correct-page head diagnostics use
known provenance and are **not** native end-to-end retrieval accuracy.

The next uncertainty is whether a learned page reader can transfer useful
information into the native answer path, rather than merely support an auxiliary
classifier. A matched random-order, multi-seed experiment should test that
connection before scaling the architecture. Relation supervision remains off by
default. All 32 trained streaming checks across these two screens matched dense
Top-1 predictions. The 38 regression tests include source-target invariance under
record swaps, unchanged answer logits when auxiliary labels are supplied, and
auxiliary gradients reaching the backbone.

Machine-readable metrics are archived under `experiments/results/`, including
both `relation-auxiliary-screen/` and `relation-auxiliary-order-pilot/`. The
manifest lists file sizes and SHA-256 hashes. Environment dependencies, page
stores and binary checkpoints remain outside Git; the continuation commands
require the named parent checkpoint in the local work directory. These synthetic
screens do not establish real-document quality, numerical Top-1 stability, or
hardware latency/memory gains.
