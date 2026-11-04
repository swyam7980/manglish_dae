"""Put .deps_mt first on the path: IndicTrans2's custom code needs transformers 4.x, the rest of the project runs 5.x.
Import this before anything that imports transformers or huggingface_hub."""
import sys

from config import WORK

FOLDER = WORK / ".deps_mt"
if FOLDER.exists() and str(FOLDER) not in sys.path:
    sys.path.insert(0, str(FOLDER))
