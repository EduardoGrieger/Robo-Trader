# utils/seed.py
from __future__ import annotations
import os, random
try:
    import numpy as np
except Exception:
    np = None
try:
    import tensorflow as tf
except Exception:
    tf = None

def set_global_seed(seed: int = 42) -> None:
    """Define seeds em random, numpy e tensorflow (se disponível) + PYTHONHASHSEED."""
    try: random.seed(seed)
    except Exception: pass
    try:
        if np is not None: np.random.seed(seed)
    except Exception: pass
    try:
        if tf is not None: tf.random.set_seed(seed)
    except Exception: pass
    os.environ["PYTHONHASHSEED"] = str(seed)

__all__ = ["set_global_seed"]
