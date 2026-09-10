"""Evict only this pinned snapshot's clean shard cache; verify with mincore."""

import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("model", type=Path)
a = p.parse_args()
files = sorted(a.model.glob("*.safetensors"))
assert len(files) == 26 and a.model.name == "995ad96eacd98c81ed38be0c5b274b04031597b0"
libc = ctypes.CDLL(None, use_errno=True)
libc.mincore.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_ubyte),
]
libc.mincore.restype = ctypes.c_int
page = os.sysconf("SC_PAGE_SIZE")


def resident(fd, size):
    n = (size + page - 1) // page
    with mmap.mmap(fd, size, access=mmap.ACCESS_COPY) as view:
        anchor = (ctypes.c_char * 1).from_buffer(view)
        vector = (ctypes.c_ubyte * n)()
        result = libc.mincore(ctypes.addressof(anchor), size, vector)
        del anchor
        if result:
            raise OSError(ctypes.get_errno(), "mincore")
        return sum(int(x) & 1 for x in vector), n


before = after = pages = total = 0
for path in files:
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        count, n = resident(fd, size)
        before += count
        pages += n
        total += size
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        count, _ = resident(fd, size)
        after += count
    finally:
        os.close(fd)
result = {
    "files": len(files),
    "bytes": total,
    "page_size": page,
    "pages": pages,
    "resident_pages_before": before,
    "resident_pages_after": after,
    "resident_fraction_after": after / pages,
    "method": "POSIX_FADV_DONTNEED per shard; mincore before/after; no global drop_caches",
}
print(json.dumps(result), flush=True)
assert after / pages < 0.001, "could not establish a cold snapshot cache"
