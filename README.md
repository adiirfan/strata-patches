# Strata patches: Radeon 890M, RTX 5080 Laptop and Radeon AI PRO R9700

Patch sets for [Strata](https://github.com/Niko1221/Strata) that run Qwen3.8-Flash-Next faster on a laptop's
integrated and discrete graphics (base commit `fb58e0d`) and on a desktop Radeon (base commit `7695859a`):

- **`strata-gfx1150-890m.patch`**: the Radeon 890M (gfx1150) integrated GPU. Faster prompt reading (about 4x) and
  writing (about 1.6x) for the Q2_0 model, all with the same output as before.
- **`strata-rtx5080-lowram.patch`**: a low-RAM mode for a GPU whose memory cannot hold the model's experts. The rest
  are read straight from one or two SSDs as they are needed, with an optional RAM cache. With IQ3_S on an
  RTX 5080 Laptop it writes about 3x faster at an 8 GB RAM cap than without it.

- **`strata-r9700.patch`**: the Radeon AI PRO R9700 (gfx1201). FP8 matrix-core kernels for a prompt's expert and dense
  products, the low-RAM mode carried forward to upstream `7695859a`, a RAM share for the experts a long prompt borrows
  from the GPU cache, fixes for GPU stalls, and conversation memory that is pinned only as far as each conversation
  reaches. A 16K prompt reads at about 3,650 tokens/s with the experts in RAM (Strata 0.1.41: about 1,500) and about
  2,775 in a 28 GB setup that runs beside ComfyUI. The 4-bit UD-Q4_K_XL model runs at the full 262K context in about
  43 GB, also beside ComfyUI.

The first two apply to a clean `fb58e0d` checkout, alone or together; the 890M results for IQ3_S and for little memory
below use both. The R9700 patch applies to `7695859a` on its own and already contains the low-RAM mode.

## Apply

```sh
git clone https://github.com/Niko1221/Strata
cd Strata
git checkout fb58e0d
git apply /path/to/strata-patches/patches/strata-gfx1150-890m.patch      # Radeon 890M
git apply /path/to/strata-patches/patches/strata-rtx5080-lowram.patch    # low-RAM mode (NVIDIA; also the 890M with IQ3_S)
```

For the R9700, on upstream `7695859a` instead:

```sh
git checkout 7695859a
git apply /path/to/strata-patches/patches/strata-r9700.patch            # Radeon AI PRO R9700 (with the low-RAM mode)
```

On a newer Strata, `git apply --3way` may resolve small differences; if a hunk still fails, check out `fb58e0d`.

Then set Strata up as usual (its `setup.py` downloads the model and builds the pack); the patches change none of
that. Build the engine with:

```sh
tools/gfx1150/build-gfx1150-sdk.sh <ROCm 10 SDK dir> <build dir> strata-890m   # Radeon 890M, AMD TheRock ROCm 10
CUDAHOSTCXX=g++-15 tools/rtx5080/build-rtx5080.sh <build dir> strata-5080      # RTX 50-series (sm_120), CUDA 13
tools/r9700/build-r9700.sh <ROCm 10 SDK dir> <build dir>                     # R9700 / RX 9070 (gfx1201), ROCm 10
```

`CUDAHOSTCXX` is only needed when the system compiler is newer than CUDA accepts.

## What the 890M patch changes

Faster kernels for the Q2_0 pack on gfx1150, all checked to give the same tokens:

- the Q2_0 gate/up experts on the fused int8 kernels;
- a rewritten expert kernel for 128-token tiles;
- padded 16-bit activations for the dense matrices;
- writing kernels with more rows per worker;
- a one-row output head for short verify windows;
- a faster GDN state commit.

It also adds `tools/gfx1150`:

- the build script;
- an example config (`strata-890m-q2_0.json`) with every switch it relies on;
- a tuned hipBLASLt table (`tools/hip/gfx1150-hipblaslt-100500.txt`);
- `draft_vocab_cut.py`, which builds the smaller draft vocabulary the config uses;
- the measuring tools.

System settings it assumes:

- **48 GB of graphics memory.** Add `ttm.pages_limit=12582912 ttm.page_pool_size=12582912` to the kernel command
  line. The default, half the RAM, cannot hold all 24,576 Q2_0 experts.
- **IOMMU off** (`amd_iommu=off`). Optional: worth 5-10% on prompts, but it disables the NPU.

For IQ3_S on the 890M, apply both patches and use `configs/strata-890m-iq3s.json`. IQ3_S's experts (50 GB) do not
fit the 48 GB, so about 73% sit in graphics memory and the rest come from the SSDs through the low-RAM path.
`--expert-cache 13800` allocated reliably beside a 262K context; with a 32K context, 15000 did. The automatic size
falls back to 30 GB when its first try at 40 GB fails.

## What the low-RAM patch changes

Everything is opt-in through environment variables. `tools/rtx5080/strata-5080-lowram-iq3s.json` sets them all.

| Setting | What it does |
| --- | --- |
| `STRATA_PROMPT_DIRECT=1` | Prompt reads go straight to the drive (O_DIRECT), not through the file cache, which is what limits them under a tight RAM cap. |
| `STRATA_DECODE_DIRECT=1` | Writing reads do the same, each layer's missing experts read in parallel by the read-ahead threads. |
| `STRATA_EXPERT_MIRROR=<file>` | A second drive serves part of every layer's experts, from a copy made by `tools/rtx5080/expert_pack.py` that stores each expert in one piece. |
| `STRATA_EXPERT_MIRROR_SHARE`, `_DECODE_SHARE` | How much of each layer the second drive serves, for prompts and for writing. |
| `STRATA_STAGE_KEEP=N` | Keeps the last N experts read in RAM. Each needs about 2 MB; on IQ3_S, use about (RAM cap in GB − 3.3) × 500. |
| `STRATA_POOL_EARLY=1` | The CPU starts on the experts already in RAM while the rest are still being read. |
| `STRATA_REFILL_BATCH=N` | Refills the GPU cache after a prompt N experts at a time. |
| `STRATA_EXPERT_SKIP=0.1` | **Changes the output.** Leaves out experts that are not at hand and weigh under 10% for every token that uses them. |

Two more settings in the config matter:

- `CUDA_MODULE_LOADING=LAZY` frees about 1 GB of RAM.
- `--kv-grow` lends the K/V memory a short conversation does not use yet to the expert cache.

Make the second drive's copy with:

```sh
python tools/rtx5080/expert_pack.py models/<model>/<shard 1>.gguf /mnt/<second drive>/experts.pack --share 0.9
```

The engine checks the copy against the model when it starts, and refuses one that does not match.

Start the server under a RAM cap with `tools/rtx5080/run-5080-lowram.sh` (`STRATA_RAM=8G` by default).

**Output.** Every setting in the table except `STRATA_EXPERT_SKIP` gives the same tokens as before. I checked this by
running a fixed continuation with `--spec-follow` and comparing the `--window-hashes` files. The GPU and the CPU round
slightly differently, though. So anything that changes which experts sit in GPU memory can change a near-tie token,
as any change of cache size does in Strata: lazy loading's spare memory, `--kv-grow`.

`STRATA_EXPERT_SKIP=0.1`, measured on the model's own story and code answers, against exact IQ3_S:

| | Same top token (story / code) | Mean KL (story / code) |
| --- | --- | --- |
| IQ3_S with `STRATA_EXPERT_SKIP=0.1` | 95.0% / 97.4% | 0.021 / 0.017 |
| Q2_0, for comparison | 82.5% / 94.7% | 0.202 / 0.052 |

## What the R9700 patch changes

Everything is opt-in, except the conversation memory below. `configs/strata-r9700-allram.json` (experts in RAM, about
55 GB), `configs/strata-r9700-28g.json` (28 GB, 256K context) and `configs/strata-r9700-q4.json` (the UD-Q4_K_XL model,
262K context, 2 slots, 15-26 GB, for a hard 28 GB limit) set it all.

| Setting | What it does |
| --- | --- |
| `STRATA_PF_FUSED_F8=1` | A prompt's expert products on the FP8 matrix cores (`v_wmma_f32_16x16x16_fp8_fp8`). Weights are decoded to E4M3 at one scale per row, activations at one scale per token, and the products add up in FP32 without the per-32-value scaling that bound the int8 kernels. Gate/up and down; `STRATA_PF_FUSED_F8D=0` keeps the int8 down. With `STRATA_PF_FUSED=1`. |
| `STRATA_PF_FP8=1` | A prompt's dense products (the GGUF's quantized attention, GDN and shared-expert weights) as FP8 hipBLASLt products, one scale per weight row. |
| `STRATA_PF_DM16=1` | The experts' outputs stored as BF16 before they are combined: half the memory traffic. |
| `STRATA_BUDGET_LEND_GIB=X` | With `--resident-budget-gib`, up to X GiB of the RAM share keep the experts a long prompt borrows from the GPU cache, so they come back from RAM, not the drive, after the prompt. 3 suits mixed prompt sizes; 6.5 favours 16K+, at a cost to 4K. |
| `STRATA_REFILL_ASYNC=1` | With the experts in RAM: the borrowed cache slots are refilled while the answer starts, not before it. |
| `STRATA_WARMUP_TOKENS=4096` | The server reads one throwaway prompt at start. Without it, the first long prompt after a start sometimes read at a fifth of the speed. |
| `STRATA_ARENA_MLOCK=1` | Locks the experts' RAM copy (needs a memory-lock limit, see below). |
| Prompt ring | With `STRATA_HIP_WMMA=1`, prompts up to 8K keep 384 experts in flight instead of 96: 4K prompts about 12% faster. `STRATA_PREFILL_RING_SMALL`, `STRATA_PREFILL_RING_SMALL_MAX` set it. |
| Conversation memory (on by default) | With `--kv-resident`, every attention layer keeps a copy of its conversation in pinned RAM. Upstream pins the full context for every layer of every session at start: 8.6 GB at 131K with `--batch 4`, mostly unused. Now each session's copy is reserved address space, pinned as its conversation grows (half as much again each time, from 8,192 tokens). A 240K conversation pins 3.8 GB; short ones a few hundred MB. `STRATA_KV_HOST_LAZY=0` pins everything up front as before; `STRATA_KV_HOST_STEP` sets the first step; `STRATA_KV_HOST_LOG=1` logs each growth. |
| Memory limit for the conversation memory | Under a memory limit (the engine's cgroup `memory.max`, else `STRATA_MEMORY_LIMIT_MIB`), a conversation copy grows only while that leaves `STRATA_KV_HOST_FLOOR_MIB` (1024) free, file cache not counted. Otherwise a request that needs more is refused, and an answer that needs more ends at its length. Pinning past the limit is not refused by the system: the engine is killed. |
| Self-restart on damaged weights | When two requests in a row produce non-finite logits, the second read from token 0 with every cached state dropped, the engine exits and the server starts it again on the next request. Upstream kept answering, and every answer after it was broken. |
| `"default_max_tokens"` in a config's `"vision"` | An image whose part does not ask for detail `"high"` is shrunk to this many image tokens before encoding, so `"max_tokens"` is reached only on request. |

Upstream switches the configs also set: `STRATA_HIP_WMMA=1 STRATA_SELECT_WMMA=1` (attention on the matrix cores),
`STRATA_PF_FUSED=1` (the fused experts for every format; the SC117 model's 20 IQ2_S layers otherwise run on MMQ),
`STRATA_PF_FUSED_KQ=1` in the Q4 config (the fused kernels for Q4_K and Q5_1 experts too: a 4K prompt's expert maths
5.9 s -> 0.6 s on the R9700),
`STRATA_ROUTE_RESIDENT=1` (about 3% faster code answers), `STRATA_UNBUFFERED_LOAD=1` and `HSA_USERPTR_FOR_PAGED_MEM=0`.

**Output.** The FP8 and BF16 settings change the numbers. On 7 questions about a 20,000-token document, answered
greedily, each step kept 6 answers word for word and reworded one summary.

**Stalls.** With the experts in RAM, the GPU's queues were suspended for 2 to 17 s per run of prompts, and up to 70 s
during loading: the kernel moved pages of the RAM copy, and the driver stops the GPU to map them again. Count it in
`/sys/class/kfd/kfd/proc/<engine pid>/stats_*/evicted_ms`. What brought it to 0.04-0.2 s per run:

- `STRATA_UNBUFFERED_LOAD=1`: the 55 GB load no longer fills the file cache.
- `HSA_USERPTR_FOR_PAGED_MEM=0`, as in upstream's report #750.
- Kernel settings: transparent huge pages `madvise`, `vm.compaction_proactiveness = 0` and
  `vm.compact_unevictable_allowed = 0`, with `STRATA_ARENA_MLOCK=1` and `LimitMEMLOCK=infinity` for the server's unit.

## Benchmarks

**Hardware:**
- CPU: AMD Ryzen AI 9 HX 370 with Radeon 890M
- RAM: 64 GB
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB, power limit 80 W
- SSDs: Samsung MZVL21T0HDLU (in a PCIe 4.0 x2 slot, about 3.6 GB/s) and Micron 2500 2 TB (PCIe 4.0 x4); the two
  together top out at about 7.1 GB/s

**Software:** Nobara Linux 44 with kernel 7.2, NVIDIA driver 595.99, CUDA 13.3, ROCm 10.2 (TheRock).

**Model:** [SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF),
Q2_0 or IQ3_S, with its MTP draft head.

**Method:** the OpenAI-compatible server, one fresh prompt per size (a unique first line, then plain text), each
followed by a 256-token story. One run per row, so expect a few percent of noise. In the RTX 5080 tables, the 8K row
is the first request after start-up, with the RAM cache still cold.

### RTX 5080 Laptop, IQ3_S, low-RAM patch, 8-bit K/V (up to 163,840 tokens)

Prompt reading, tokens/s:

| Context | 8 GB | 16 GB | 24 GB | 32 GB | 40 GB | All experts in RAM (~52 GB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 900 | 998 | 990 | 998 | 992 | 1,219 |
| 16K | 1,221 | 1,320 | 1,445 | 1,548 | 1,551 | 1,468 |
| 32K | 1,109 | 1,204 | 1,310 | 1,400 | 1,424 | 1,450 |
| 64K | 1,313 | 1,405 | 1,487 | 1,551 | 1,576 | 1,468 |
| 128K | 1,214 | 1,306 | 1,387 | 1,455 | 1,487 | 1,403 |
| 151K | 1,218 | 1,302 | 1,394 | 1,455 | 1,480 | 1,398 |

Writing, tokens/s:

| Context | 8 GB | 16 GB | 24 GB | 32 GB | 40 GB | All experts in RAM (~52 GB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 30.8 | 32.5 | 36.4 | 33.8 | 32.9 | 45.7 |
| 16K | 33.5 | 39.8 | 49.9 | 53.6 | 49.1 | 53.0 |
| 32K | 31.5 | 42.5 | 48.0 | 57.5 | 56.5 | 52.1 |
| 64K | 30.4 | 41.1 | 49.4 | 53.9 | 54.7 | 47.7 |
| 128K | 27.1 | 36.6 | 43.5 | 49.3 | 52.1 | 48.9 |
| 151K | 28.4 | 37.2 | 46.5 | 48.7 | 49.7 | 48.2 |

The RAM cap is a cgroup limit on the whole server, file cache included, with swap off.

- **Low-RAM columns:** `tools/rtx5080/strata-5080-lowram-iq3s.json`, with `STRATA_STAGE_KEEP` sized to each cap
  (2600, 6300, 10300, 14300, 18300).
- **Last column:** Strata's normal mode (every expert copied into RAM), with the same 8-bit K/V settings.

The 8-bit K/V stops at about 160K tokens on a 16 GB card. Beyond that it no longer fits beside the experts the
prompt path borrows.

### Radeon 890M, Q2_0, 890M patch (all 24,576 experts in graphics memory)

| Context | Prompt, tokens/s | Writing, tokens/s |
| ---: | ---: | ---: |
| 8K | 566 | 21.0 |
| 16K | 607 | 22.1 |
| 32K | 573 | 18.9 |
| 64K | 578 | 20.5 |
| 128K | 554 | 18.8 |
| 254K | 494 | 18.7 |

With Strata's own community benchmark script (`bench/results/2026-09-30-community-rtx-5090/benchmark.py`,
unchanged), median prompt / writing tokens/s:

| Prompt tokens | Prompt | Writing |
| ---: | ---: | ---: |
| 4,096 | 570.5 | 25.30 |
| 32,768 | 554.5 | 24.79 |
| 128,000 | 520.4 | 23.41 |
| 260,000 | 468.1 | 22.69 |

Without the patch (stock Fedora ROCm 7.1 build), a 6K-token prompt read at 130 tokens/s and a story wrote at
15.6 tokens/s; with it, about 580 and 25.

### Radeon 890M, IQ3_S, both patches (about 73% of experts in graphics memory)

| Context | Prompt, tokens/s | Writing, tokens/s |
| ---: | ---: | ---: |
| 8K | 447 | 14.0 |
| 16K | 531 | 13.6 |
| 32K | 508 | 14.4 |
| 64K | 520 | 13.4 |
| 128K | 497 | 13.9 |
| 249K | 452 | 13.0 |

These used `configs/strata-890m-iq3s.json` with `--max-context 262144` and the second-drive copy. With the 890M patch
alone and the automatic cache size (30 GB), the same server read 8K at 234 tokens/s and 32K at 125, and wrote at
9.5-11.

### Radeon 890M with little memory, both patches

The 890M has no memory of its own, so its expert cache comes out of the laptop's RAM. With both patches,
`--expert-cache` can be set far below the whole model: the experts that do not fit are read from the SSDs as they are
needed, as on the RTX 5080. These runs find how small the cache can go and still be usable, for example on battery.

**Setup:**
- `configs/strata-890m-q2_0-lowram.json` for Q2_0, and `configs/strata-890m-iq3s.json` for IQ3_S, each with the
  `--expert-cache` shown in the tables.
- 32K context, 8-bit K/V, and the second-drive copy (`--share 0.9`).
- The server started with the engine's own RAM capped at 4 GB. Copy the config into the Strata folder, then:
  `STRATA_CONFIG=strata-890m-q2_0-lowram.json STRATA_RAM=4G tools/rtx5080/run-5080-lowram.sh`

On a native pack, `--expert-cache N` is a byte budget of N times the largest expert. IQ3_S's experts vary in size, so
it fits about 30% more experts than N.

**Memory** is the whole laptop's: the drop in available memory at the server's peak. The 890M's share of RAM does not
count towards the 4 GB cap, so the cap alone would understate it.

**Tests:**
- a 6K-token prompt;
- a 256-token story after a 2K-token prompt;
- a 1,024-token code answer (a Python module with its tests, temperature 0) through the server.

One run per row, on the laptop's 240 W charger.

Q2_0, tokens/s:

| Total memory | `--expert-cache` | 6K prompt | Story | Code answer |
| ---: | ---: | ---: | ---: | ---: |
| ~9 GB | 1,100 | 367 | 9.6 | 11.1 |
| ~13 GB | 4,200 | 490 | 12.1 | 15.1 |
| ~21 GB | 10,400 | 536 | 13.9 | 20.1 |
| ~29 GB* | 16,500 | 571 | 17.5 | – |
| ~40 GB | 24,576 (all) | ~580 | 24.9 | 31.8 |

\* Estimated from the cache size, not measured.

IQ3_S, tokens/s:

| Total memory | `--expert-cache` (experts held) | 6K prompt | Story | Code answer |
| ---: | ---: | ---: | ---: | ---: |
| ~10 GB | 560 (718) | 222 | 7.7 | 8.8 |
| ~14 GB | 2,130 (2,753) | 420 | 9.0 | – |
| ~18 GB | 3,700 (4,817) | 436 | 10.1 | 12.6 |
| ~22 GB | 5,280 (6,899) | 453 | 11.1 | – |
| ~30 GB | 8,350 (10,908) | 481 | 14.4 | 17.4 |
| ~47 GB | 15,000 (19,523) | 519 | 16.7 | 20.0 |

What they show:
- **Q2_0 at about 13 GB** is the smallest setup that still feels usable: 15 tokens/s on code, and a 6K prompt read in
  about 12 seconds. Below it, writing drops to 10-11 tokens/s and prompts slow by a quarter.
- **IQ3_S needs roughly 10 GB more** than Q2_0 for the same writing speed. Its experts are about 1.5 times bigger, so
  every token reads more from the drives. Its prompts suffer most at the bottom: 222 tokens/s at 10 GB.
- **Above about 20 GB, prompts gain little:** 8-15% more all the way to the full model. Writing keeps improving, and
  jumps once every expert fits (Q2_0: 17.5 → 24.9 tokens/s from about 29 to 40 GB).

**Power.** The same code answer on a 70 W USB-C charger, with the processor package's power (CPU and 890M together)
averaged while it wrote:

| Model, total memory | Code answer, tokens/s | Package power | Energy per token |
| --- | ---: | ---: | ---: |
| Q2_0, ~9 GB | 10.5 | 45.9 W | 4.4 J |
| IQ3_S, ~19 GB | 11.7 | 35.2 W | 3.0 J |
| IQ3_S, ~30 GB | 15.9 | 40.0 W | 2.5 J |
| IQ3_S, ~48 GB | 18.2 | 42.3 W | 2.3 J |

- **The smallest cache costs the most energy per token.** It reads the drives the most and runs the longest. A
  bigger cache costs memory but saves battery.
- **`STRATA_POOL_SPIN_US=0`** (a stock Strata setting) stops the CPU workers from spinning while they wait for work.
  On IQ3_S at about 19 GB it cut package power from 35.2 W to 28.8 W, for 11.3 → 11.0 tokens/s. The low-memory config
  sets it.
- **Let a discrete GPU switch off.** On this laptop the RTX 5080, kept awake but idle, drew about 18 W, which counts
  against the battery whatever the 890M is doing.

These 890M power figures were read from the processor, not the battery, so they do not show whether the battery
charged. The RTX 5080 figures below do.

### Low power: RTX 5080 Laptop on a USB-C charger

The fastest settings found that keep the battery charging while the server works. IQ3_S with the low-RAM patch,
32 GB of RAM, 8-bit K/V, screen off.

**Settings:**
- **Power saver, with the processor's turbo off** (power saver alone leaves it on):
  ```sh
  powerprofilesctl set power-saver
  echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
  ```
- **Card limit 55 W**, the lowest this laptop accepts (asus-armoury `nv_tgp`; `nvidia-smi -pl` is not supported).
- **Card memory at 9,001 MHz:** `sudo nvidia-smi -lmc 405,9001`. The 55 W covers the card's memory too, so slower memory
  leaves more for its core: about 40% faster prompts than at full memory speed.
- **On 70 W only, the card's core capped at 525 MHz:** `sudo nvidia-smi -lgc 210,525`.
- `STRATA_POOL_SPIN_US=0`, as in the example config.

**100 W charger** (20 V × 5 A):

| Request | Prompt, tokens/s | Writing, tokens/s | Battery |
| --- | ---: | ---: | ---: |
| Code answer | – | 45.5 | +0.5 W |
| 16K | 1,234 | 40.8 | +4.7 W |
| 32K | 1,132 | 41.1 | +10.3 W |
| 64K | 1,238 | 39.6 | +7.6 W |

**70 W charger** (20 V × 3.5 A):

| Request | Prompt, tokens/s | Writing, tokens/s | Battery |
| --- | ---: | ---: | ---: |
| Code answer | – | 31.0 | +0.1 W |
| 8K | 636 | 22.5 | ±0 W |
| 16K | 727 | 25.6 | +2.6 W |
| 32K | 687 | 26.9 | +1.3 W |
| 64K | 719 | 26.7 | +1.3 W |
| 128K | 678 | 22.9 | +1.6 W |
| 151K | 671 | 25.1 | ±0 W |

**Notes:**
- **Battery** is the slope of the battery's charge counter over each request (+ charging); a full battery cannot be
  read, so these runs started below full. The code answer is a 1,024-token Python answer, temperature 0.
- **On power saver the processor varies from run to run**, so writing can come out lower (down to about half) at the
  same settings.
- **Stopping the server mid-answer** has left the card at full clock (about 55 W) with no program on it until another
  program opened the card: check `nvidia-smi` after a stop.

### Radeon AI PRO R9700, IQ3_S, R9700 patch

- **Hardware:** AMD Ryzen 5 7500F (6 cores), 96 GB DDR5-5600, Radeon AI PRO R9700 32 GB at a 230 W power limit
  (default 300 W), Patriot Viper VP4300L 2 TB (about 3.55 GB/s).
- **Software:** Debian 13 with kernel 7.1.8, ROCm 10.2.0a20261009 (TheRock), Strata `7695859a` with the patch.
- **Method:** as above, after the start-up warm-up. One run per row; a range is two or three runs.

Prompt reading, tokens/s:

| Setup | RAM | 4K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: |
| Strata 0.1.41 + low-RAM patch (before) | 24 GB | – | – | 1,302 |
| Strata 0.1.41, experts in RAM (before) | 77 GB | – | – | 1,515 |
| `strata-r9700-allram.json` | about 55 GB | 2,850-2,890 | 3,600-3,680 | 3,300-3,750 |
| `--resident-budget-gib 29`, `STRATA_BUDGET_LEND_GIB=7` | 36 GB | 2,652 | 3,469-3,482 | 2,911-3,306 |
| `strata-r9700-28g.json` (256K context) | 28 GB | 1,812-2,221 | 2,762-2,786 | 2,424 |
| `--resident-budget-gib 17`, `STRATA_BUDGET_LEND_GIB=3`, 128K context | 24 GB | 1,604-1,857 | 2,686-2,722 | 2,391 |

Writing after each prompt: 60-71 tokens/s in every setup. A 1,024-token code answer: about 100 tokens/s with the
experts in RAM, 82-94 in 28 GB.

Long context, 24 GB, 256K context (`--resident-budget-gib 14.5`: the K/V takes 3.1 GB of RAM at 256K):

| Prompt | Prompt, tokens/s | Writing, tokens/s |
| --- | ---: | ---: |
| 64K | 2,951 | 63.9 |
| 128K | 2,784 | 61.0 |
| 240K (242,537 tokens) | 2,736 | 59.6 |

**Beside ComfyUI** (`strata-r9700-28g.json`; ComfyUI renders MiniMax H3 on another card, an RTX 3080):

| | Alone | Both at once |
| --- | ---: | ---: |
| Short H3 render | 53.85 s | 54.0 s |
| Long H3 clip (it asks Strata for its frame prompts) | 596-703 s before | 591 s |
| Strata 16K prompt, tokens/s | about 2,780 | 2,677-2,807 |
| Strata code answer, tokens/s | 82 | 82-94 |

The lowest available RAM during the long clip was 15.1 GiB of 94.

### Radeon AI PRO R9700, UD-Q4_K_XL, R9700 patch

- **Model:** [huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF),
  `unsloth-UD-Q4_K_XL` (111 GB), packed with `tools/iq_pack.py --compat-bf16`, with the same MTP draft head. Images
  use the base model's `mmproj-F16.gguf`.
- **Setup:** 262,144-token context, 8-bit K/V with 32,768 tokens on the GPU. The table below: a 30 GiB RAM share for
  the experts, 4 parallel slots, no memory cap; the engine's peak was 42.8 GB. `configs/strata-r9700-q4.json` is the
  28 GB setup further down.
- **Hardware and method:** as for IQ3_S above, on the running server. One run per row.

| Prompt | Prompt, tokens/s | Writing, tokens/s |
| --- | ---: | ---: |
| 1K | 242 | 53.9 |
| 4K | 736 | 54.8 |
| 16K | 1,885 | 52.6 |
| 32K | 1,849 | 54.3 |
| 64K | 2,432 | 56.9 |
| 128K | 2,178 | 53.4 |
| 240K (242,574 tokens) | 2,212 | 50.1 |

**Less RAM.** The conversation memory now grows with each conversation, and the RAM it no longer holds goes to the
experts. 131K context, `STRATA_PF_FUSED_KQ=1`, under a memory cap:

| RAM cap | Experts in RAM | Conversation memory | 4K | 16K | Peak |
| --- | ---: | --- | ---: | ---: | ---: |
| 32 GB | 15 GiB | pinned up front (before) | 572 | 1,730 | 31.3 GB |
| 32 GB | 20 GiB | grows | 644 | 1,876 | 28.5 GB |
| 32 GB | 22 GiB | grows | 674 | 1,896 | 31.4 GB |

At 20 GiB, a 96K prompt read at 1,923 tokens/s and the peak rose to 30.0 GB. Under a cap the share must leave room for
the conversations to grow: a cap reached by the engine coincided with non-finite logits twice.

**Beside ComfyUI** (MiniMax H3 on an RTX 3080, one part of a long clip):

| | Alone | Both at once |
| --- | ---: | ---: |
| H3 part render | 768 s | 632 s |
| 16K prompt, tokens/s | 1,885 | 1,852 |
| 240K prompt, tokens/s | 2,212 | 2,148 |
| Writing, tokens/s | 50-57 | 49-55 |

The lowest available RAM was 26 GB of 94, with no swapping. Under a 28 GB cap with a 17 GiB share and 131K context it
also ran beside a render: 4K 578, 16K 1,728, 64K 2,150 tokens/s.

**A hard 28 GB limit.** `configs/strata-r9700-q4.json` runs the server in a systemd scope with `MemoryMax=28G`,
`MemoryHigh=27G` and no swap, with a 6.5 GiB RAM share for the experts and 2 slots. The share is small so that the
worst case fits: the main session and both slots each holding a full 262K conversation (about 3.6 GB each). Three
conversations of about 242,000 tokens held at once took 25.1 GB (peak 25.5); with nothing long held, 15.1 GB.

| Prompt | Prompt, tokens/s | Writing, tokens/s |
| --- | ---: | ---: |
| 1K | 153 | 31.4 |
| 4K | 455 | 48.1 |
| 16K | 1,511 | 50.1 |
| 64K | 2,128 | 51.2 |
| 242K | 1,764 | – |

The same limit with a 13 GiB share and 4 slots (which fits only while the conversations held stay under about
400,000 tokens in all), beside an H3 render:

| | Result |
| --- | ---: |
| 16K prompt, tokens/s | 1,647 |
| 240K prompt, tokens/s | 1,917 |
| Writing, tokens/s | about 43 |
| Highest RAM | 24.5 GB |
| H3 part render | 761 s (768 alone) |

That run had 4 slots; the config has 2, which gives the GPU's expert cache about 2 GB more and uses a little less RAM
(21.7 GB after a 5,000-token answer). Keep the engine clear of the limit itself: every time it reached a hard limit in
these tests (at start-up under a 22 GB cap, or pinning the full context up front under 32 GB), its output came out
non-finite. `MemoryHigh` a little below `MemoryMax` has the kernel drop the scope's file cache first.

**Known issue.** About one start in ten, the first long prompt with this model gave non-finite logits, and every
request after it did too until a restart. It was seen with and without the conversation-memory change, never in 12
attempts to cause it (busy drive, a render starting, `STRATA_PF_FUSED_KQ` on and off), and never with IQ3_S. The cause
is not known; the self-restart above limits it to one failed request.

## License

The patches modify Strata and are under its MIT license (`LICENSE`).
