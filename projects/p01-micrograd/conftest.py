"""Makes `import micrograd` and `import expressions` work without installing anything."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT / "src", ROOT, ROOT / "tests"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
