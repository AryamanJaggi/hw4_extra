#!/usr/bin/env bash
# Build the CPU backend on Windows with MinGW g++ (no MSVC / CMake needed).
# Usage: bash build_cpu_windows.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYBASE=$($PY -c "import sys; print(sys.base_prefix)")
PYLIB=$($PY -c "import sys; print('python%d%d' % sys.version_info[:2])")
PBINC=$($PY -m pybind11 --includes | tr ' ' '\n' | grep pybind11 | sed 's/^-I//')
EXT=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT=python/needle/backend_ndarray/ndarray_backend_cpu$EXT
g++ -O3 -march=native -shared -std=c++14 \
  -I"$PYINC" -I"$PBINC" src/ndarray_backend_cpu.cc \
  "$PYBASE/$PYLIB.dll" -static-libgcc -static-libstdc++ -static \
  -o "$OUT"
echo "built $OUT"
