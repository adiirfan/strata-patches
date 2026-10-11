# Strata patches: Radeon 890M and RTX 5080 Laptop

Two patch sets for [Strata](https://github.com/Niko1221/Strata) (base commit `fb58e0d`) that run Qwen3.8-Flash-Next
faster on two kinds of laptop graphics:

- **`strata-gfx1150-890m.patch`**: the Radeon 890M (gfx1150) integrated GPU. Faster prompt reading (about 4x) and
  writing (about 1.6x) for the Q2_0 model, all with the same output as before.
- **`strata-rtx5080-lowram.patch`**: a low-RAM mode for a GPU whose memory cannot hold the model's experts. The rest
  are read straight from one or two SSDs as they are needed, with an optional RAM cache. With IQ3_S on an
  RTX 5080 Laptop it writes about 3x faster at an 8 GB RAM cap than without it.

Both apply to a clean Strata checkout, alone or together. The 890M results for IQ3_S and for little memory below
use both.

## Apply

```sh
git clone https://github.com/Niko1221/Strata
cd Strata
git checkout fb58e0d
git apply /path/to/strata-patches/patches/strata-gfx1150-890m.patch      # Radeon 890M
git apply /path/to/strata-patches/patches/strata-rtx5080-lowram.patch    # low-RAM mode (NVIDIA; also the 890M with IQ3_S)
```

On a newer Strata, `git apply --3way` may resolve small differences; if a hunk still fails, check out `fb58e0d`.

Then set Strata up as usual (its `setup.py` downloads the model and builds the pack); the patches change none of
that. Build the engine with:

```sh
tools/gfx1150/build-gfx1150-sdk.sh <ROCm 10 SDK dir> <build dir> strata-890m   # Radeon 890M, AMD TheRock ROCm 10
CUDAHOSTCXX=g++-15 tools/rtx5080/build-rtx5080.sh <build dir> strata-5080      # RTX 50-series (sm_120), CUDA 13
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

## License

The patches modify Strata and are under its MIT license (`LICENSE`).
