# MoE cache + TurboQuant integration notes

Branch: `integrate/moe-turbo`. Base: this fork (mainline llama.cpp with the `--moe-cache-mib` expert cache).
TurboQuant source: TheTom/llama-cpp-turboquant `feature/turboquant-kv-cache` at `643956891`.

## Commits

| Commit | Content |
|---|---|
| `ac0f82e65` | INVESTIGATION.md |
| `fead2f482` | TurboQuant KV types, TQ3_1S/TQ4_1S weight types, `TURBO_WHT` op (CPU) |
| `9583f1f99` | CUDA quantize/dequantize, set_rows, `TURBO_WHT` kernels |
| `83d437187` | Build fixes for the two commits above |
| `79d59a974` | CUDA flash attention vec kernels for turbo K/V |
| `3ff38e962` | llama: turbo KV cache (head padding, WHT on Q and FA output, FA requirement) |
| `39d9565c9` | `-ctk/-ctv` and llama-bench accept turbo types |

## Build

The documented command is `cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release`. What was actually used:

```
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
ninja -C build llama-server llama-cli llama-perplexity llama-bench
```

- Ninja: CUDA 13.3 has no Visual Studio 2026 integration, so the VS generator cannot build CUDA.
- Arch 86: `native` found no GPU at configure time.
- MSVC 14.51 (VS 2026 Community) and Windows SDK 10.0.26100 were put on `INCLUDE`/`LIB`/`PATH` by hand instead of vcvars.

## CLI

- New cache types for `-ctk`/`-ctv` (and llama-bench): `turbo2`, `turbo3`, `turbo4`, `turbo5`, `turbo6`.
- Turbo K or V requires `-fa on`. With `-fa auto` it is enabled automatically. With `-fa off` context creation fails.
- MoE flags: `-cmoe`, `-ncmoe`, `--moe-cache-mib`, `--moe-cache-cpu-share F|auto` (auto searches 0-60% while generating), `--moe-cache-prefetch` (predicts the next layer's experts with its router and uploads the misses ahead on a separate copy stream, ubatches of up to 4 tokens, off by default).
- generel's `--moe-expert-cache-size` is not present (see benchmark gate). No flag collisions.

## How the caches interact

- There is one MoE expert cache: `--moe-cache-mib`. It works on the experts kept on the CPU by `-cmoe`/`-ncmoe`.
- TurboQuant only changes the KV cache. It does not touch expert weights or the MoE cache.
- They share the same GPU memory. VRAM saved by turbo KV can be given to `--moe-cache-mib`, which is where the decode gain comes from on 8 GB cards.

## Smoke tests

Model: `Sage-A1.5-Q4_K_M.gguf`, qwen35moe 35B-A3B, 256 experts (8 used), head dim 256, 20.2 GiB. RTX 3070 Ti 8 GB.

Arguments from `SageQ4.ps1`, host changed to 127.0.0.1 and port to 8091:
`-ngl all -fit off -b 8192 -ub 512 -np 1 -t 15 -tb 15 -fa on -kvo --load-mode mlock --cache-ram 0 --jinja -c 131072 --mmproj ... --mmproj-device none`

Request: 1528 prompt tokens, 128 generated, `/completion`. Peak VRAM is the total GPU use from nvidia-smi (about 690 MiB was in use before the runs).

| Run | Build | KV | MoE | Prompt tok/s | Decode tok/s | Peak VRAM MiB |
|---|---|---|---|---|---|---|
| a | this | q8_0 / q8_0 | `-cmoe` | 755 | 34.2 | 4420 |
| b | this | turbo3 / turbo4 | `-cmoe` | 746 | 34.4 | 3559 |
| c | this | turbo3 / turbo4 | `-cmoe --moe-cache-mib 3000` | 766, 783 | 50.5, 50.5 | 6577 |
| d | this | q8_0 / q8_0 | `-cmoe --moe-cache-mib 2800` | 786, 805 | 48.1, 50.5 | 7161 |
| e | this | turbo3 / turbo4 | `-cmoe --moe-cache-mib 3600` | 729, 763 | 51.7, 55.5 | 7177 |
| g | generel (`I:\llama\3070\llama.cpp-moe-cache`) | q8_0 / q8_0 | `--moe-expert-cache-size 40` | 828, 841 | 51.8, 52.0 | 7255 |

Runs with two numbers sent the request twice. The second run is warmer.

Turbo output was coherent in b, c and e.

## Perplexity

`llama-perplexity -c 512 --chunks 40 -fa on -cmoe --moe-cache-mib 3000`. The text is a concatenation of `docs/build.md`, `docs/function-calling.md`, `docs/multimodal.md` and `tools/server/README.md` (213 KB). This is not wikitext.

| KV | PPL |
|---|---|
| q8_0 / q8_0 | 4.0528 +/- 0.0948 |
| turbo3 / turbo4 | 4.0621 +/- 0.0950 |

## Benchmark gate

The rule was to port generel's cache only if it wins decode by about 25% or more at equal VRAM.

- generel (g) at 7255 MiB: 52.0 tok/s. This build with q8_0 (d) at 7161 MiB: 50.5 tok/s. That is about +3% for generel.
- This build with turbo3/turbo4 (e) at 7177 MiB: 55.5 tok/s, about +7% over generel.

generel's cache was not ported.

## Deviations from the plan

- turbo5/turbo6 were kept as cache types. Since `1c5c53da9` they have FA vec instances for head dims 128 and 256.
- The 21 turbo FA vec instances are always compiled, independent of `GGML_CUDA_FA_QUANTS`.
- `ggml/src/ggml-turbo-quant.c` was copied verbatim and still contains the fork's comments.

## Not ported

- TQ3_1S / TQ4_1S CUDA matmul (mmvq, MMQ tiles, `common.cuh` traits) and the quantize ftypes 43/44. The types exist in ggml but run on the CPU only.
- Fused turbo MMA decode kernels (`fattn-mma-turbo.cuh`, fattn-mma-f16 changes), KV streams, sparse mask, D512 and HIP instances.
- Auto-asymmetric K upgrade, layer-adaptive mode. InnerQ is wired since `23a72ba2b` (opt-in with `TURBO_INNERQ=<tokens>`, applied inside TURBO_WHT, single GPU).
- CPU flash attention tiling changes.

## Not verified

- turbo2 and symmetric turbo3/turbo3 or turbo4/turbo4. turbo5/turbo6 were checked with perplexity only.
- Models with head dims that are not a multiple of 128 (the padding path).
- Long-context quality (perplexity ran at 512 ctx only), multi-GPU, `-ncmoe` with partial layers, `--moe-cache-cpu-share`.
- test-backend-ops was not run for the new types.

## Follow-up items (no benchmarks yet)

| Item | Result |
|---|---|
| Per-layer cache budgets | Already present: the slots of a group are shared by all of its layers |
| Frequency-aware eviction | Already present: decayed use counts in 8 classes, victim by age/2^class |
| Prefetch next layer's experts | `--moe-cache-prefetch`, `eb7ff69d6`. Greedy output identical with and without. 63% of prefetched experts were used in one test |
| Adaptive CPU share | `--moe-cache-cpu-share auto`, `b60236f4b` |
| CUDA graphs with the cache | Already working: graphs are reused for every split, nothing to fix |
| Fused turbo MMA kernels | Not ported: 37 conflicts with the current MMA kernel, and the fork has no turbo3-K/turbo4-V instance |
| turbo5/turbo6 FA kernels | `1c5c53da9` |
| MTP speculative decoding | Already supported: `--spec-type draft-mtp`, the Sage GGUF has an MTP layer. Works with turbo KV, the cache and prefetch |
| TQ3_1S/TQ4_1S CUDA matmul | Not ported: TQ4_1S is 5.0 bpw, TQ3_1S 4.0 bpw, IQ4_XS/IQ3 give the same size gain with full CUDA support |
| InnerQ | `23a72ba2b`, opt-in, perplexity within noise on Sage |

Perplexity, 12 chunks of 512 on the same text: turbo3/turbo4 5.4989, turbo5/turbo6 5.4966, turbo6/turbo6 5.4995, turbo3/turbo4 with `TURBO_INNERQ=2048` 5.5224 (all +/- 0.24).
