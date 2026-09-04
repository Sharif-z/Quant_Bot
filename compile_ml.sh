#!/bin/bash
echo "=========================================="
echo " ONYX CUSTOM C++ ML ENGINE COMPILER"
echo "=========================================="

echo "[1] Checking for clang++ compiler..."
if ! command -v clang++ &> /dev/null
then
    echo "clang++ not found. Please run: pkg install clang"
    exit 1
fi

echo "[2] Compiling onyx_ml.cpp into shared object (libonyx_ml.so)..."
# -O3 tells clang to aggressively optimize for speed
# -shared -fPIC creates a Position Independent Executable library for ctypes
clang++ -shared -fPIC -O3 onyx/ml/engine/onyx_ml.cpp -o onyx/ml/engine/libonyx_ml.so

if [ $? -eq 0 ]; then
    echo "[3] SUCCESS: Compiled to onyx/ml/engine/libonyx_ml.so"
    echo "Python ctypes can now bind to the engine."
else
    echo "[3] FAILED: Compilation error."
fi
