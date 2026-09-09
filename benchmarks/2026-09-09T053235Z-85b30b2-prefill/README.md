# Prefill after GDN and resource-lifetime changes

Source `85b30b2`, sp10 GB10, pinned 35B BF16 snapshot, FP32 recurrence,
1024 prompt IDs, first measured prefill after two warmups. The asset download
was active, so use the separately recorded core runs for latency comparisons.

The capture has 1204 kernels, 533.572 ms summed kernel time and 545.954 ms
first-to-last-kernel span. GPU work covers 543.586 ms; 2.367 ms has no GPU work.
Pinned-host allocation/free time is now zero, compared with 40.320 ms before
execution resources survived prefix rebuilds.

| Kernel category | Calls | Summed milliseconds |
| --- | ---: | ---: |
| Grouped MoE projections | 80 | 312.996 |
| Warp GDN recurrence, FP32 state | 30 | 132.562 |
| Parallel convolution outputs | 30 | 6.445 |

MoE projections account for 58.7% of summed kernel time and recurrence 24.8%.
The earlier serial-GDN trace had 962.632 ms recurrence and 497.233 ms convolution.
The extra 30 kernels are final convolution-state writebacks. Kernel sums are
work attribution, not independent end-to-end savings.

This confirms that prefill is now dominated by MoE and recurrence GPU work,
while retaining provider resources removed the observed pinned allocation/free
cost. Graph capture primarily targets decode gaps; it cannot remove most of this
prefill compute. Full traces remain in `.prototypes/profiles/` and
`sp10@sp10:~/qs3-profiles/` under this capture name.
