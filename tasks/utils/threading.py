import os
from contextlib import contextmanager

@contextmanager
def single_threaded():
    original_env = {k: os.environ.get(k) for k in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    ]}
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        yield
    finally:
        for k, v in original_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                del os.environ[k]