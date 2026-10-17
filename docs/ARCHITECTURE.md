# Architecture and behavior

`decoder_llm.model` retains the completed project's state-dict layout. Tokens and
learned absolute positions are summed, then passed through dropout, a stack of
pre-normalized residual blocks, final RMSNorm, and a vocabulary projection tied
to the token embedding matrix. Each block has RMSNorm -> causal multi-head
self-attention -> residual, then RMSNorm -> GELU MLP -> residual. Q/K/V share a
linear projection. Attention scores use a lower-triangular boolean mask and
softmax with training dropout. Position, width, head count, layers, MLP width,
dropout, bias, vocabulary size and optional separator are configurable.

The explicit attention implementation favors inspectability. It does not use a
fused attention kernel. Its training attention matrix has quadratic sequence
memory. RMSNorm accumulates half/bfloat16 squares in float32 to avoid overflow;
training itself currently uses ordinary float32.

## Data and loss

`CharTokenizer` constructs a sorted character vocabulary from training data only.
Serialized vocabularies restore their exact order. Unknown characters fail with
a descriptive error; there is no silent unknown-token substitution. Characters
are Unicode code points, not bytes or language-aware subwords.

`CharacterDataset` samples prebatched random next-character windows. A private
generator supports exact replay, includes the final valid start, and permits a
corpus of exactly `context_length + 1` characters. Use `batch_size=None` and
`num_workers=0` with a DataLoader; multiple workers are explicitly rejected to
avoid duplicate random streams.

The historical `<HUMAN>: ... <ASSISTANT>: ...` format selects paired training.
Exact duplicate pairs are removed before the ordered split; externally supplied
validation pairs are checked for exact overlap. Each row becomes
`prompt + newline + target`. Prompt predictions and padding use target `-100`.
Long prompts are left-truncated enough to keep the separator and at least the
first supervised target. All-masked losses are rejected. The first vocabulary
character supplies padding inputs, but causality and masked labels prevent
padding positions from changing supervised predictions. No padding/EOS token is
learned. A newline separator is appended automatically during paired generation;
disable this when sending an already formatted prefix.

Raw text is split at a character boundary when no separate validation file is
provided. This is reproducible but may cut sentences and does not remove repeated
documents. Provide independently curated held-out files for meaningful work.
Validation characters must be present in training. Neither random windows nor
the included synthetic text establish real-world quality or translation ability.

## Generation and cache semantics

`generate` accepts `(batch, time)` long tensors and returns prompt plus generated
tokens, including any appended separator. Greedy and sample strategies support
batched input, temperature, top-k, top-p, optional EOS, repetition penalties, and KV caching. Completed
batch rows receive EOS until every row finishes. An empty prompt is rejected.
The caller's train/eval mode is restored, including after a forward failure.
Optional repetition penalties apply to characters already seen in each prompt
and continuation: positive logits are divided by the penalty, negative logits
are multiplied by it. This happens before temperature/top-k/top-p filtering and
also applies to beam search. The default `1.0` preserves existing behavior.
Prompt dtype and vocabulary bounds are checked even for zero-token requests.

Each cache layer stores `(K, V)` tensors shaped `(batch, heads, time, head_dim)`.
Chunked decoding offsets both positions and the causal mask by the cache length.
Cache layers must agree on length, batch/head dimensions, device and dtype. Input
plus cache cannot exceed the configured context length. Caches are caller-owned
and should be used for inference; the model does not carry a hidden global cache.

When the context fills, generation crops the prefix to the last context window
and recomputes it. Simply dropping the oldest keys would keep stale learned
absolute positions and disagree with full-window inference. This correct reset
reduces cache speedups during long runs. Cache tensors use concatenation rather
than preallocated buffers, which also limits performance.

Beam search retains the original full-prefix approach and supports one sequence.
It ranks cumulative log probability divided by generated length to the configured
length penalty. It now handles EOS and beam widths larger than the vocabulary.
`use_cache` does not change beam search. The Python function preserves historical
beam-search defaults; the new CLI explicitly defaults to greedy decoding.

## Training, checkpoints and evaluation

`run_training` uses AdamW, explicit learning rate/weight decay, clipping with
nonfinite-gradient rejection, and a fixed step budget. Validation sampling has a
separate reset seed; loss is weighted by the number of nonignored target tokens.
Reports store full configurations, corpus/split hashes, seed, runtime versions,
Git revision/dirty status and per-step metrics without copying raw corpus text.

Version 1 checkpoints contain state dictionaries, configuration dictionaries,
ordered character lists, optimizer state, training/data RNG states and metadata.
Writes replace the destination atomically. Restricted `weights_only=True` loading
is explicit. Resume verifies configuration, data and runtime identity. It restores
optimizer and RNG state before the next batch. CPU exact-resume equality is
covered by a regression test; identical results across PyTorch versions, devices,
CUDA libraries or hardware are not promised. Package dependencies require
PyTorch >=2.10, addressing known restricted-loader vulnerabilities; see
[the upstream advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
Treat external checkpoints as untrusted even with a restricted loader.

Historical checkpoints pickle a config object. They are rejected by normal
loading. `convert-legacy --trust-pickle` is a deliberately explicit local migration
tool for checkpoints you created and trust: general pickle loading can execute
code. Conversion keeps the model weights and exact character order, but cannot
reconstruct optimizer/RNG state missing from old files.

Held-out text evaluation scores every next-character target once in nonoverlapping
chunks, resetting context between chunks and including the final short chunk.
Perplexity is the exponential of token-weighted cross entropy under that policy.
Paired evaluation preserves the historical character BLEU/ROUGE-L functions, uses
a fixed generation budget independent of reference length, reports skipped OOV
prompts and fails if none can be evaluated. These character metrics are not
standard corpus BLEU or a claim of translation quality.
