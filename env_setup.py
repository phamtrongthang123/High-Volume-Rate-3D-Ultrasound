"""Shared environment setup for all scripts in this project.

Sets ZEA_CACHE_DIR to a local cache folder within the project directory,
so cached models and datasets are self-contained and portable.

Usage: import env_setup  (before importing zea)
"""

import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _SCRIPT_DIR / "cache"
_CACHE_DIR.mkdir(exist_ok=True)

os.environ.setdefault("ZEA_CACHE_DIR", str(_CACHE_DIR))
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
