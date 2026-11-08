# Task: faster MoE-cache decode, lossless only

You are continuing performance work on a llama.cpp fork, branch `integrate/moe-turbo`. Read `INTEGRATION_NOTES.md` and `docs/moe-cache.md` first.

There are two tasks:

- **Task A:** the partial prefetch wait.
- **Task B:** removing the GPU bubble at the scheduler's expert-copy callback.

The goal is higher decode tok/s with **bit-identical output**.

## Hard rules

- **Lossless only.**
  - Every change must give byte-identical output for the seeded run below, and an unchanged PPL (5.9741).
  - No smaller quants, no expert skipping, no approximations.
  - `--moe-cache-skip` exists but is lossy: don't use it or tune it.
- **Keep `-ub 1024`.**
- **Git:**
  - Never touch `master`; never force-push; never push without asking the owner.
  - Commit each verified feature with a clear message, and no AI or assistant attribution of any kind.
- **No code comments of any kind.** Don't edit existing comments.
- **No Python and no helper scripts.** Use the shell, the build tools and the binaries.
- **Temp files:**
  - Never redirect to `/dev/null`, `NUL` or `nul`; use named files in `$TEMP`.
  - Logs go to `$TEMP`; truncate long outputs.
- **Scope:** don't probe the PC, network or personal files outside the project and the model paths below.
- **Tests and failures:**
  - Give the owner a list of tests or profiling runs for approval before running anything beyond a single A/B per change.
  - If a step fails twice, stop and report.
- **Reporting:** report results faithfully, including no-gain results.
- **Cleanup:** remove all temporary profiling code before committing.

## Hardware and model

- **Hardware:** Windows 11, RTX 3070 Ti 8 GB. The PCIe link is about 32 GB/s; check it with `nvidia-smi dmon -s t -d 1`.
- **Model:** `I:/models/Sage-A1.5-Q4_K_M.gguf` (qwen35moe 35B-A3B, 40 layers, 256 experts, 8 used per token).
  - The expert weights live in pinned host memory (`CUDA_Host`) and are cached on the GPU by the MoE cache (`src/llama-moe-cache.cpp`).

## Build

```
M="C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231"; S="C:/Program Files (x86)/Windows Kits/10"; V=10.0.26100.0
export INCLUDE="$M/include;$S/Include/$V/ucrt;$S/Include/$V/shared;$S/Include/$V/um;$S/Include/$V/winrt"
export LIB="$M/lib/x64;$S/Lib/$V/ucrt/x64;$S/Lib/$V/um/x64"
export PATH="/c/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64:/c/Program Files (x86)/Windows Kits/10/bin/$V/x64:/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja:$PATH"
ninja -C build -j 12 llama-server llama-cli llama-perplexity llama-bench llama-completion
```

If linking fails with LNK1104, a running process holds `llama.dll`. Ask the owner to close it.

## Tests

### Decode A/B (one per change)

Run from `build/bin`. The baseline is about 73.4 tok/s.

```
./llama-completion.exe -m I:/models/Sage-A1.5-Q4_K_M.gguf -ngl all -fit off -b 8192 -ub 1024 -t 15 -tb 15 -fa on -ctk turbo3 -ctv turbo4 -cmoe --moe-cache-mib 4000 -kvo --load-mode mlock -c 8192 --temp 0.7 --top-k 20 --top-p 0.9 -s 42 -no-cnv -n 600 --moe-cache-prefetch -p "Write a long, highly descriptive historical essay on the evolution of naval warfare." > "$TEMP/run.log" 2>&1
grep "  eval time" "$TEMP/run.log"
grep -vE "^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ " "$TEMP/run.log" > "$TEMP/run.txt"
```

Save `$TEMP/run.txt` from the unchanged build as the reference. Every change must pass `cmp` against it.

### Perplexity (final check)

`$TEMP/ppl.txt` is the owner's text file. The result must be `PPL = 5.9741`.

```
./llama-perplexity.exe -m I:/models/Sage-A1.5-Q4_K_M.gguf -ngl all -fit off -t 15 -tb 15 -fa on -ctk turbo3 -ctv turbo4 -cmoe --moe-cache-mib 4000 -kvo --load-mode mlock -f "$TEMP/ppl.txt" -c 512 --chunks 8 -b 512 -ub 512 --moe-cache-prefetch
```

### Server speedtest

The owner runs this.

```
./llama-server.exe -m I:/models/Sage-A1.5-Q4_K_M.gguf --mmproj I:/Models/Qwen/mmproj-F16.gguf --mmproj-device none --host 127.0.0.1 --port 8091 -ngl all -fit off -b 8192 -ub 1024 -np 1 -t 15 -tb 15 -fa on -ctk turbo3 -ctv turbo4 -cmoe --moe-cache-mib 4000 -kvo --load-mode mlock --cache-ram 0 --jinja --temp 0.7 --top-k 20 --top-p 0.9 -c 131072 --moe-cache-prefetch
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "I:/PiAgent/speedtest.ps1" -BaseUrl "http://127.0.0.1:8091"
```

## What is known (measured)

### Per-token budget

- **Overall:** about 13.6 ms per token (73.4 tok/s). GPU compute alone is about 8 ms.
- **Prefetch:** the predicted experts of layer il+1 are uploaded on a separate copy stream while layer il computes.
  - A worker thread issues the copies (`run_prefetch` / `prefetch_worker` / `submit_prefetch`).
  - Per layer: about 6.3 copies of ~630 KiB, about 200 µs at ~20 GB/s. Per token that's about 8 ms of link time.
  - Only about 60% of the prefetched experts are used (stats: `pf_used` / `pf_uploaded`).
- **Main-path misses:** about 1.6 copies per layer, about 53 µs. Per token that's about 2.1 ms, on the compute stream and serial with compute.
- **Link usage:** about 10 of the 13.6 ms per token. Decode is bounded by bus usage and by the overlap between transfers and compute, not by GPU compute.
- **Copy size:** each copy is one expert of one tensor, about 630 KiB. Raising per-copy bandwidth from 20 to 32 GB/s is worth at most about 1 ms per token.

### Already tried with no gain (don't repeat)

- Pinned slot maps.
- Moving `wait_prefetch(...)` in `copy()` from the top of the function to just before the upload loop: identical output, but no speed change.
- Prefill streaming (`--moe-cache-stream`): it works and is lossless, but prefill didn't get faster.

### Prefill

- About 920 tok/s for 4129 tokens.
- It's dominated by uploads of about 68 GB at 15 to 30 GB/s.
- It is not part of these tasks.

## Task A: partial prefetch wait

- **Where:** `src/llama-moe-cache.cpp`, functions `prefetch_next`, `run_prefetch`, `wait_prefetch` and `copy`.
- **The problem:**
  - `prefetch_next(il)` builds one job for layer il+1, and the job records one event (`device::copy_event`) at its end.
  - `copy()` for layer il+1 calls `wait_prefetch`, which makes the compute stream wait for the whole job, even when the layer only needs some of the prefetched experts.
- **The change:**
  - Order the prefetch fills by predicted confidence; check what `gp.lru.prefetch(...)` returns and how `pred` is ranked.
  - Split the job into a few chunks, and record an event after each chunk on the copy stream. This needs a small fixed array of events per device, created and freed next to `copy_event`.
  - In `copy()`, once the real ids are known, make the compute stream wait only for the event of the last chunk that holds an expert the layer actually uses.
  - The rest of the copies finish in the background.
- **Correctness: no slot may be read or overwritten while a copy into it is still in flight.**
  1. An unused prefetched expert whose copy is still running may be evicted by `plan(...)` and rewritten by a main-path upload. That upload is on another stream, so it needs the full wait or a different slot.
  2. The next `prefetch_next` must not write into slots that are still in flight. The copy stream is in order, so that holds on its own.
  3. Whatever the slot map points at must be complete before the expert matmul runs.
  - The simplest safe rule: if the plan touches any slot whose chunk isn't covered yet, do the full wait.
- **Expected gain:** unknown, perhaps a few percent. Measure it with one decode A/B.

## Task B: GPU bubble at the expert-copy callback

- **Where:** `ggml/src/ggml-backend.cpp`, `ggml_backend_sched_copy_input` (about line 2095) and `ggml_backend_sched_compute_splits` (about line 2200).
- **The problem:**
  - The host expert weights are split inputs. Before the copy callback (`sched->callback_copy`, set via `ggml_backend_sched_set_copy_callback`), the scheduler waits for the split backend to finish using the input copy.
  - With one graph copy there are no events (`sched->events[...][cur_copy] == NULL`), so it calls `ggml_backend_synchronize(split_backend)` and the GPU drains at every layer.
  - The callback (`llama_moe_cache::impl::copy`, reached through `llama_context::sched_copy_experts` in `src/llama-context.cpp`) then reads the selected expert ids back with `ggml_backend_tensor_get_async` + `ggml_backend_synchronize`. It plans, enqueues the miss uploads and the slot-map upload, and submits the next prefetch.
  - The GPU is idle from the end of the router until the next compute is enqueued, 40 times per token. That's about 140 µs per layer of non-compute time, including about 53 µs of serial uploads.
- **Things to explore:**
  1. The synchronize before the callback looks redundant when the callback synchronizes itself. Check whether it can be skipped, or replaced by an event wait for host-weight inputs that the callback handles. Keep the fallback path correct when the callback returns false.
  2. Find graph work in the layer split that doesn't depend on the selected experts and enqueue it before the callback blocks. For example, qwen35moe has a shared expert.
  3. Anything else that shortens the drain, launch, readback and enqueue chain. Nsight Systems, if installed, is the best tool. Otherwise use temporary `ggml_time_us` timers, removed before committing.
- **Constraints:** the scheduler change must not break other backends or the CPU fallback, and the decode output must stay bit-identical.

## Deliverables

- One commit per verified, kept change.
- A short report for each change with the before/after tok/s and the `cmp` result, plus the final PPL.
- Update `INTEGRATION_NOTES.md` and `docs/moe-cache.md` for the kept changes.
- Don't push; the owner decides.
