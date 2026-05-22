"""Runtime storage settings for Windows machines with small C: drives."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / ".tmp"


def configure_runtime_storage() -> Path:
    """Route Python and library temporary/cache files into the project drive."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    temp_dir = RUNTIME_DIR / "temp"
    mpl_dir = RUNTIME_DIR / "matplotlib"
    joblib_dir = RUNTIME_DIR / "joblib"
    pycache_dir = RUNTIME_DIR / "pycache"
    for path in (temp_dir, mpl_dir, joblib_dir, pycache_dir):
        path.mkdir(parents=True, exist_ok=True)

    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(temp_dir)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(joblib_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_dir)
    sys.pycache_prefix = str(pycache_dir)
    return RUNTIME_DIR
