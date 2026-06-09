"""Fraud detection platform package.

macOS note: PyTorch and XGBoost each ship their own OpenMP runtime. When both are active
and multithreaded, fitting XGBoost can segfault (duplicate OpenMP runtimes). Pinning OpenMP
to a single thread before either library loads avoids this. It must be set here, at package
import, because the OpenMP runtime reads these env vars when it first initialises (on torch
import) — setting them later has no effect. The data here is small, so single-threaded
training costs little. Remove these on Linux/containers if you want full multithreading.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
