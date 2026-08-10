"""Single source of truth for where bot_processors/ keeps its non-code data
and cache files (reference JSON/xlsx harvested from Agmarknet, geocoding
caches, the market-price table's legacy JSON export, etc.).

Every module that reads/writes one of these files imports DATA_DIR from
here instead of computing its own Path(__file__).parent — before this
project was reorganized into pricing/location/calls/voice/rag/core
subpackages, every such module lived directly in bot_processors/ alongside
its data file, so a bare Path(__file__).parent "just worked". Once modules
moved a directory deeper, each one recomputing that path independently
would mean N separate places that all have to agree on how many parent
directories to walk up — one mistake in any of them silently points at
the wrong (nonexistent, or worse, a *different* file's) path. Computing it
once here removes that whole class of mistake.
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
