# Shared build utilities

Parse compiler inputs with field-type and unknown-field checks:

```python
from qsutil.config import CudaTarget, TritonSpec, parse

spec = parse(spec_json, TritonSpec)
target = parse(target_json, CudaTarget)
```

Errors propagate directly from JSON decoding and dacite.
