# Integration investigation: MoE expert cache + TurboQuant KV on mainline

Date: 2026-11-08. Base: `master` = `upstream/master` = `372dc5ca1` (no fast-forward needed).

## 1. Remotes and branches

| Remote | URL | Branch inspected | Tip |
| --- | --- | --- | --- |
| upstream | ggml-org/llama.cpp | master | 372dc5ca1 |
| generel | GenerelSchwerz/llama.cpp | moe-cache | f3f31a788 |
| generel | GenerelSchwerz/llama.cpp | codex/port-draft-controls, codex/port-draft-residency | de3662aa2, d74ed5093 |
| turbo | TheTom/llama-cpp-turboquant | feature/turboquant-kv-cache | (fetched tip) |

`moe-cache-drafting` does not exist on GenerelSchwerz. The two closest branches are `codex/port-draft-controls` (4 commits) and `codex/port-draft-residency` (5 commits). Both are fully contained in `generel/moe-cache`, so they need no separate handling.

## 2. Fork branch shape

| Branch | Merge-base with upstream | Behind upstream | Ahead (non-merge) | Files changed | Lines |
| --- | --- | --- | --- | --- | --- |
| generel/moe-cache | c35b66744 (2026-10-08) | 689 | 367 | 290 | +134,096 / -4,659 |
| turbo/feature/turboquant-kv-cache | 15586e2d7 (2026-08-06) | 1901 | 490 (90 merges) | 409 | +80,542 / -3,780 |

`git cherry` shows that almost no commit in either branch is patch-equivalent to an upstream commit. The exceptions are 1 turbo commit by patch-id and 3 by identical subject: TENSOR_READ_LAZY #27794, reasoning_effort, and Spark2_5 #27868. Neither branch is a thin feature branch.

### 2.1 generel/moe-cache commits by feature

Authors: Rocco A 233, Anbeeld 82, anirbankar 26, GenerelSchwerz 8, others. 11 commits are reverts of experiments.

| Group | ~Commits | Content |
| --- | --- | --- |
| A. Original slot-pool expert cache | 28 | `CUDA_MoE_Cached` buffer type, per-tensor LRU slot pools, async H2D, `--moe-expert-cache-size`, sibling prefetch, staging when n_unique > slots |
| B. Grouped MoE decode / graph plans | ~70 | grouped Q4_K / Q4_0 / NVFP4 decode, typed graph preflight, plan caching and "certification", CUDA graph capture of grouped decode |
| C. Prefill staging / overflow | ~45 | split MoE MMQ dispatch, overflow admission, staging waves, mapped MMQ padding |
| D. Host memory / placement / fit | ~26 | pinned / pageable host banks, `--moe-expert-cache-host-pinned-mb`, `--moe-expert-cache-mib`, `--moe-expert-cache-layers`, joint fit admission (`common/fit.cpp` +1.6k) |
| E. Hybrid CPU/GPU source executor + profiles | ~17 | `--moe-hybrid`, `--moe-expert-profile`, `--moe-profile-*`, `--moe-gpu-miss-*`, CPU reference kernels (AVX2/AVX512) |
| F. Early router | ~23 | `--moe-early-router`, device-side routing, segmented TopK |
| G. Speculative / draft / MTP | ~30 | independent draft ubatch, draft KV residency, partial GPU KV residency, MTP workspace sharing |
| H. Misc kernels / server / sampling | rest | compact causal masks, live workspace sizing, dense penalty counts, server cancellation |
| I. Docs / tests / CLAUDE.md / README | ~60 | fork README rewrite, wiki pointers, hiring/Ko-fi notes |

Hot files: `ggml-cuda/moe-cache.cu` +22k (new), `ggml-cuda/ggml-cuda.cu` 16.5k changed lines, `src/llama-context.cpp` +4.3k, `ggml-cuda/moe-source-core.cu` +4.2k, `moe-fidelity-executor.cu` +3.3k, `ggml-cpu/ggml-cpu.cpp` +2.6k, `ggml-cuda/norm.cu` +2.3k, `ggml-backend.cpp` +2.2k, `src/llama-model.cpp` +1.9k, `common/fit.cpp` +1.6k, `mmvf.cu` +1.6k, `src/llama-graph.cpp` +1.2k, `tools/server/server-context.cpp` +1k, `mmq.cu` +0.9k, `common/arg.cpp` +0.5k.

### 2.2 turbo/feature/turboquant-kv-cache commits by feature

The branch was rebased on 2026-07-31 as squashed "port" commits by jabbatheduck. Those commits are the clean spine of TurboQuant:

| Commit | Content | Size |
| --- | --- | --- |
| 00fda770b | ggml core types + CPU kernels | 9 files, +1.2k |
| 5fd308947 | CUDA MMVQ / WHT / inner-quant kernels, sparse-V in fattn-vec | 63 files, +4.4k |
| c3a048776 | SYCL | 33 files |
| 2a716ac47 | Vulkan / Metal / HIP | 25 files, +12.7k |
| 1f7946026 | llama KV cache, rotation data, models, common | 41 files, +5.4k |
| 273bcc630, e130aef60 | parity / audit fixes, TQ4_1S dp4a | 13 files |

After the spine there are about 480 more commits from about 30 contributors:

| Group | ~Commits | Unique to fork? |
| --- | --- | --- |
| T1. TurboQuant KV types turbo2/3/4 (+ turbo5/6 added in PR #381) | ~60 | unique |
| T2. TQ3_1S / TQ4_1S weight types, MMVQ-TQ, fused TQ paths | ~25 | unique |
| T3. `GGML_OP_TURBO_WHT` (128-group signed WHT), InnerQ, rotation data | ~12 | partly upstream (see 3.2) |
| T4. Sparse-V dequant skip in `fattn-vec.cuh` (`TURBO_SPARSE_V`, default on, threshold 1e-6) | 2 | unique |
| T5. Its own adaptive MoE cache (`--moe-cache MODE`, default `auto`; CUDA / HIP / Metal / Vulkan providers; giveen / leloch) | ~96 | competes with upstream |
| T6. Metal / Vulkan / SYCL / HIP ports | ~90 | out of scope (CUDA build) |
| T7. Models: qwen4exp, muse-glimmer, spark2-5, dflash, eagle3, laguna, DeepSeek-V4 split, drama3 | ~60 | mostly upstream now (add/add conflicts) |
| T8. ConvRot weights (Q8_CR / Q5_CR / Q6_CR) | ~20 | unique, out of scope |
| T9. KV-stream / phase arena, gdn-replay, TurboAnchorKV example, speculative-adaptive | ~40 | unique, out of scope |
| T10. CI / docs / AGENTS.md | ~40 | drop |

New TurboQuant-only files: `ggml/src/ggml-turbo-quant.c`, `ggml-cuda/{turbo-quant.cuh, turbo-wht.cu/.cuh, turbo-innerq.cu/.cuh, mmvq-tq.cu/.cuh, fattn-mma-turbo.cuh}`, 61 FA template instances, `src/turbo-rotation-data.h`, `src/turbo-rotation-data-32.h`. TurboQuant hunks in shared files: `ggml.h`, `ggml.c`, `ggml-cpu/{ggml-cpu.c, ops.cpp, quants}`, `ggml-cuda/{fattn.cu, fattn-common.cuh, fattn-vec.cuh, fattn-mma-f16.cuh, set-rows.cu, convert.cu, mmvq.cu, ggml-cuda.cu}`, `src/llama-kv-cache.*`, `src/llama-graph.cpp`, `src/llama-context.cpp`, `common/arg.cpp`, `common/common.*`, `gguf-py/gguf/constants.py`, `tools/llama-bench`, `tools/quantize`.

## 3. What upstream already has

### 3.1 MoE cache (#29887 and follow-ups)

Upstream `src/llama-moe-cache.cpp` is a llama-level device cache for experts kept in host memory. Flags: `--moe-cache-mib N` and `--moe-cache-cpu-share F`. API: `llama_context_params.moe_cache_size`, `.moe_cache_cpu_share`.

| Upstream PR | Date | In generel base? | In turbo base? |
| --- | --- | --- | --- |
| #29887 add GPU cache for MoE experts | 2026-10-07 | yes | no |
| #30112 multi-GPU MoE cache | 2026-10-08 | yes | no |
| #30637 SLRU 50/50 | 2026-10-19 | no | no |
| #30966 run missed experts on CPU (`--moe-cache-cpu-share`) | later | no | no |
| #31233 LRFU/MQ hybrid ClassLRU | later | no | no |

generel merged upstream on 2026-10-08 and keeps both caches. In its `llama-context.cpp`, it throws `"--moe-cache-mib cannot be combined with --moe-expert-cache-size"`. Its cache is a separate ggml-cuda buffer-type design and does not touch `src/llama-moe-cache.cpp`.

### 3.2 Hadamard KV rotation (#21038) and WHT kernels

Upstream rotates K and V with a Hadamard `mul_mat` for every quantized KV type when head_dim % 64 == 0. The CUDA fast path is `ggml-cuda/fwht.cu` via `GGML_HINT_SRC0_IS_HADAMARD`, with follow-ups #29100 (>512 blocks), #29096 (F16 input), #31073 (constants on GPU), #31406 (shared-KV layers), and #28498 (saved rotation metadata). The off switch is `LLAMA_ATTN_ROT_DISABLE`.

The turbo fork contains #21038 but turns it off globally by default. It brings rotation back per side with `LLAMA_ATTN_ROT_K_OVERRIDE` / `LLAMA_ATTN_ROT_V_OVERRIDE`. Its own `GGML_OP_TURBO_WHT` is a separate op: a 128-element group with random sign vectors on both sides, fused into set-rows and FA dequant.

### 3.3 Feature status table

| Feature | Fork | Status vs upstream | Plan |
| --- | --- | --- | --- |
| Expert cache in VRAM for host experts (basic LRU) | generel A | **already upstream** (`--moe-cache-mib`, now SLRU / ClassLRU + CPU miss share) | use mainline |
| Multi-GPU MoE cache | generel | already upstream (#30112) | n/a (single GPU) |
| Grouped MoE decode (Q4_K / Q4_0 / NVFP4), graph-captured | generel B | **unique** | gate on benchmark |
| Cached prefill staging / overflow waves | generel C | unique (mainline has CPU-share instead) | gate on benchmark |
| Host pinning budgets / placement fit | generel D | partly upstream (`common/fit` exists, no MoE-aware joint fit) | only with B |
| Hybrid source executor, profiles, early router | generel E/F | unique, experimental | skip |
| Draft ubatch / draft KV residency / partial KV residency | generel G | unique | skip (not requested) |
| turbo2/3/4 KV types (+5/6) | turbo T1 | **unique** | port |
| TQ3_1S / TQ4_1S weights | turbo T2 | **unique** | port (CPU + CUDA) |
| WHT rotation | turbo T3 | **partly upstream**: the concept and FWHT kernel exist; the signed 128-group TURBO_WHT op and codebooks do not | port the turbo op as-is; do not replace with upstream FWHT in this pass |
| Sparse-V | turbo T4 | **unique** | port |
| Adaptive MoE cache `--moe-cache` | turbo T5 | **duplicates upstream** + a third cache design | drop |
| New models (qwen4exp, muse-glimmer, spark2-5, DSV4 split, ...) | turbo T7 | **already upstream** (add/add conflicts) | drop, keep mainline |
| ConvRot, KV-stream, gdn-replay, TurboAnchorKV, Metal / Vulkan / SYCL | turbo T6/T8/T9 | unique, out of scope | drop |

## 4. Conflict prediction

Dry runs used `git merge --no-commit --no-ff` on a detached HEAD at `upstream/master`, followed by `git merge --abort`. The fork-vs-fork check used `git merge-tree`. `master` was never checked out during the merges.

| Merge | Conflicted files |
| --- | --- |
| upstream/master + generel/moe-cache | 71 (1 modify/delete: `tests/test-recurrent-state-rollback.cpp`) |
| upstream/master + turbo branch | 138 (modify/delete on Metal shader and workflows; add/add on qwen4exp, muse-glimmer, memory-hybrid-idx, kv-cache-dsa-iswa) |
| generel/moe-cache + turbo branch | 137 |

Conflict hunks in the focus files (count of `<<<<<<<` markers):

| File | generel vs upstream | turbo vs upstream | generel vs turbo |
| --- | ---: | ---: | --- |
| src/llama-context.cpp | 17 | 47 | conflict |
| src/llama-graph.cpp | 28 | 12 | conflict |
| src/llama-model.cpp | 3 | 27 | conflict |
| src/llama-kv-cache.cpp / .h | 3 / 2 | 18 / 4 | conflict |
| ggml-cuda/ggml-cuda.cu | 15 | 9 | conflict |
| ggml-cuda/fattn-mma-f16.cuh | 0 | 37 | conflict |
| ggml-cuda/fattn-common.cuh / fattn.cu | 3 / 1 | 8 / 6 | conflict |
| ggml-cuda/fattn-vec.cuh | 0 | 1 | conflict |
| ggml-cuda/mmq.cuh / mmq.cu / mmq-load-tiles.cuh | 20 / 14 / 26 | 3 / 2 / 0 | conflict |
| ggml-cuda/mmvq.cu | 0 | 11 | conflict |
| ggml-cuda/set-rows.cu | 0 | 0 (auto-merged) | clean |
| ggml-cuda/moe-cache.cu / .cuh | n/a | n/a | **add/add: two unrelated files with the same name** |
| ggml-backend.cpp | 1 | 2 | conflict |
| common/arg.cpp | 2 | 5 | conflict |
| include/llama.h | 0 | 8 | conflict |
| ggml/include/ggml.h | 1 | 4 | conflict |

Other large ones: generel has tests/test-backend-ops.cpp 10, server-context.cpp 8, llama-graph.h 7, norm.cu 6. turbo has qwen4exp.cpp 64, gguf constants.py 36, memory-hybrid-idx.cpp 25, test-backend-ops.cpp 22, dflash.cpp 20, speculative.cpp 19.

### 4.1 Semantic conflicts (no textual marker, but break things)

1. **ggml_type IDs collide.** Upstream has `GGML_TYPE_F8_E4M3 = 43` and `COUNT = 44`. Turbo uses `TURBO2_0 = 43`, `TURBO3_0 = 44`, `TQ3_1S = 45`, `TQ4_1S = 46`, `TURBO4_0 = 47`, `Q*_CR = 48..50`, `TURBO5_0 = 51`, `TURBO6_0 = 52`. TQ3_1S and TQ4_1S are stored in GGUF files, so they must stay 45 and 46 to load existing TQ models. The turbo KV types exist only at runtime and can move.
2. **llama ftype collides**: upstream `MOSTLY_F8_E4M3 = 42` vs turbo `MOSTLY_Q8_CR = 42`. The collision goes away when ConvRot is dropped. Turbo's `MOSTLY_TQ3_1S = 43` and `MOSTLY_TQ4_1S = 44` are free upstream.
3. **Double rotation.** Upstream applies the #21038 Hadamard rotation to every `ggml_is_quantized` KV type, which includes the turbo types. Turbo avoids this by disabling #21038 globally. That would also change mainline behavior for q8_0/q4_0 KV.
4. **Three MoE caches.** upstream `--moe-cache-mib`, generel `--moe-expert-cache-*`, turbo `--moe-cache`. Each hooks `mul_mat_id` and expert buffer placement differently.
5. **Generel bypasses upstream's newer cache work.** It sits on the Oct 8 cache and has no SLRU, ClassLRU, or CPU share.

## 5. Is GenerelSchwerz's cache still worth porting?

The plain caching (group A) is not worth porting: mainline now does the same job with a better eviction policy plus CPU-side miss compute. The only possible value is group B, grouped CUDA decode with graph capture, plus the prefill staging in group C. The fork reports about 2.6x decode on Qwen3.6-35B, but the comparison is against "stock llama.cpp" on a 16 GB card. The README does not say whether stock included `--moe-cache-mib`, and those are historical numbers, not the current head.

Porting cost is very high. It is about 134k lines, including a 22k-line `moe-cache.cu`, 16.5k changed lines in `ggml-cuda.cu`, and changes to the CPU backend, the allocator, and the scheduler. It has 71 conflicting files against upstream and 689 upstream commits to absorb, and it overlaps turbo in FA / MMQ / MMVQ / ggml-cuda.cu. Cherry-picking is not practical: the 367 commits interleave experiments and reverts. Only a true merge is realistic.

**Recommendation: don't port it by default. Decide with a benchmark first.** Build `generel/moe-cache` as-is in a separate worktree (no integration) and compare it with this mainline build on the same 8 GB GPU and the same Qwen3.6-35B-A3B Q4_K_M. Use matched VRAM budgets: `--moe-expert-cache-mib X` vs `--moe-cache-mib X --moe-cache-cpu-share` (default and tuned). Port only if generel wins decode by at least about 25% at equal peak VRAM.

## 6. Recommended integration order and approach

Branch: `integrate/moe-turbo` from `master`.

**Step 1: TurboQuant, ported onto mainline (not merged).** Merging the turbo branch would pull in 1901 commits of divergence, DSV4, ConvRot, Metal / Vulkan / SYCL, a third MoE cache, and duplicate models: 138 conflicts for mostly unwanted code. Port only T1–T4, for CPU + CUDA. Commit groups:

1. `ggml`: type enums and traits, `ggml-turbo-quant.c`, `GGML_OP_TURBO_WHT`, CPU quant/dequant/vec_dot, gguf-py constants. IDs: TQ3_1S = 45 and TQ4_1S = 46 (unchanged, GGUF-compatible); TURBO3_0 = 44, TURBO4_0 = 47, TURBO2_0 = 48 (moved off F8_E4M3's 43); TURBO5_0 = 49 and TURBO6_0 = 50 only if wanted; `GGML_TYPE_COUNT` updated. Turbo-built GGUFs with TQ weights still load. Saved KV state files from the turbo fork will not, which is expected for runtime types.
2. `ggml-cuda`: turbo-quant.cuh, turbo-wht, turbo-innerq, set-rows turbo path, convert, MMVQ-TQ, `ggml-cuda.cu` dispatch / supports_op.
3. `ggml-cuda` FA: fattn-common / fattn-vec turbo dequant + sparse-V, fattn.cu dispatch, `fattn-mma-turbo.cuh`, template instances (CMake glob). I recommend turbo2/3/4 only, plus the q8_0 / f16 pairings, to limit compile time.
4. `llama`: KV cache type handling, rotation data, graph WHT insertion, layer-adaptive / auto-asymmetric env knobs, `llama-context` validation. The rotation policy is resolved as: **upstream #21038 stays on for non-turbo quantized KV (mainline default preserved) and is forced off for turbo KV types**, which carry their own WHT. `LLAMA_ATTN_ROT_DISABLE` keeps its meaning.
5. `common` / tools: `-ctk` / `-ctv` accept turbo2/3/4, llama-bench and quantize tables, tests (test-turbo-quant, backend-ops cases).

What I bring from the turbo branch is its final, debugged version of each hunk, not the original port commits. Each commit message names the source commits.

**Step 2: build and smoke-test** turbo3 with mainline `--moe-cache-mib`. This is already a complete deliverable: one build with a MoE expert cache and turbo KV.

**Step 3: benchmark gate** for generel (section 5). Stop and report the numbers.

**Step 4, only if the gate passes:** run `git merge generel/moe-cache` into `integrate/moe-turbo`. That is one merge commit, and its message lists the conflict groups: CUDA MMQ/MMID, FA (vs turbo), llama graph / context / model, common/arg + fit, server, tests. Excluded and kept as mainline: README, CLAUDE.md / AGENTS.md, docs/fork-*, media. Upstream `--moe-cache-mib` and generel `--moe-expert-cache-*` stay mutually exclusive, using the error generel already has, and are documented. Turbo's `--moe-cache` is never brought in, so its flag name can't collide. If no flag is given, no cache runs, the same as mainline.

## 7. Risks / open questions for approval

- Approve dropping the turbo fork's own MoE cache, ConvRot, Metal / Vulkan / SYCL, KV-stream, and model additions.
- Approve the type-ID remap (TURBO2_0 -> 48) and the rotation policy (turbo-only off).
- turbo5/6: include or skip? Recommended: skip, which saves about 30 FA template instances of compile time.
- Approve the benchmark gate rather than porting generel unconditionally.
- Smoke-test model path: global memory says models live in `D:\Ai\Qwen`. I need the exact GGUF filename.
