"""CageMetrics: reproducible UFC performance forecasts."""

import os

# Set native defaults before NumPy/scikit-learn are imported. OpenMP's setting
# is thread-local; applying a limit only in ASGI startup misses request threads.
# Explicit operator settings still take precedence.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.3.0"
