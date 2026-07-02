from __future__ import annotations

if __package__:
    from . import main
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qwen36_vectors import main


if __name__ == "__main__":
    main()
