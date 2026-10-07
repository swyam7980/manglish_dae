import os
import sys
import tempfile
from pathlib import Path

# Isolate all data/output paths before any project module imports config.
_WORK = Path(tempfile.mkdtemp(prefix="mdae_test_"))
os.environ["MDAE_WORKDIR"] = str(_WORK)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
