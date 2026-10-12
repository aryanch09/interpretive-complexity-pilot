"""Test package for the interpretive-complexity pilot.

Run with either:
    python3 -m pytest tests/ -q
    python3 -m tests.test_truncation_contrast
    python3 -m tests.test_trend_tests

TensorFlow guard
----------------
TensorFlow 2.21.0 is installed on this machine and its abseil C++ layer takes a
global mutex on import that is never released under Python 3.13 + PyTorch 2.12,
hanging the process with no error (see CLAUDE.md, "TensorFlow / Abseil Mutex
Deadlock").  `analysis/embeddings.py` guards itself, but a test module that
imports any part of the pipeline could reach `transformers` by another path, so
the guard is installed here too, before anything else in the package imports.

`sys.modules['tensorflow'] = None` makes `import tensorflow` raise ImportError
rather than load the library.  Nothing in these tests needs TensorFlow.
"""

import sys

sys.modules.setdefault("tensorflow", None)
