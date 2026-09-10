# Full-snapshot weight loading on sp10

Source `89a264e`, pinned 35B BF16 snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, GB10/CUDA 13.0. Storage is ext4 on
`/dev/nvme0n1p2`, Samsung MZALC4T0HBL1-00B07 NVMe; see `storage.txt`.
All five prescribed-script tests pass: 723 allocations, 64.560 GiB file payload,
0.469 MiB generated zero-fill, and exact planned byte/allocation counts.

| Cache condition | Backend/order | Backend setup s | Load s | Setup + load s |
| --- | --- | ---: | ---: | ---: |
| Uncontrolled | Managed first | 0.331 | 38.299 | 38.630 |
| Uncontrolled | Pinned | 1.474 | 57.330 | 58.803 |
| Uncontrolled | Managed repeat | 0.252 | 44.474 | 44.726 |
| Verified cold | Managed | 0.299 | 63.043 | 63.342 |
| Verified cold | Pinned | 1.554 | 56.913 | 58.467 |

The uncontrolled sequence starts with substantially different global cache
sizes; it cannot choose a backend. The cold pair uses `POSIX_FADV_DONTNEED` only
on the pinned snapshot's 26 shards, then verifies **zero resident pages** with
`mincore` before each test. The script does not clear global caches. Snapshot
files are not modified. Header planning follows eviction and precedes timing.
These are cold Linux file-cache runs, not claims about SSD-controller cache.

Pinned staging uses four 1 GiB buffers and 693 transfers. In the cold run its
phases are 2.087 s allocation, 0.404 s event waits, 54.408 s reads and 0.008 s
copy enqueue; ring initialization adds 1.554 s. Enqueue time is not DMA duration.
Managed spends 63.035 s in direct reads into final managed allocations. That
measurement includes any destination-page preparation caused by the read; it
does not isolate storage speed from memory handling.

Pinned setup-plus-load is 7.7% shorter in this cold pair. This completes the
previously interrupted four-buffer comparison and demonstrates why cached small
probes do not predict full-load time. It is one controlled pair, not an exhaustive
tuning result. The tests do not measure first GPU access or steady inference from
pinned/device versus managed weights. Keep that comparison as the next backend
decision gate; no production backend default changes here.

Reproduction, with the prescribed gitignored script at this revision:

```sh
./run_cuda_test.sh just weight-loader-compare
./run_cuda_test.sh just weight-loader-cold-bench /home/sp10/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
```

`weight-loader-compare` runs the ignored managed, pinned and managed tests sequentially;
`weight-loader-cold-bench` prepares the shard cache before each managed/pinned test. Both invoke
`cargo test --lib bench_real_qwen36_bf16_<backend>_load -- --ignored --nocapture
--test-threads=1`. Raw logs, residency records and extracted `results.json` are
included. `collect.py` regenerates that JSON from the logs.
