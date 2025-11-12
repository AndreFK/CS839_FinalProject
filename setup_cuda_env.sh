#!/bin/bash
# Script to set up CUDA library paths for TensorFlow in WSL2

# Get the venv path
VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/venv"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Get library paths from nvidia packages
CUDA_RUNTIME_LIB=$(python -c "import nvidia.cuda_runtime.lib; import os; print(os.path.dirname(nvidia.cuda_runtime.lib.__file__))" 2>/dev/null)
CUBLAS_LIB=$(python -c "import nvidia.cublas.lib; import os; print(os.path.dirname(nvidia.cublas.lib.__file__))" 2>/dev/null)
CUDNN_LIB=$(python -c "import nvidia.cudnn.lib; import os; print(os.path.dirname(nvidia.cudnn.lib.__file__))" 2>/dev/null)

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CUDA_RUNTIME_LIB:$CUBLAS_LIB:$CUDNN_LIB:$LD_LIBRARY_PATH"

# Also set CUDA paths
export CUDA_HOME="/usr/local/cuda-11.8" 2>/dev/null || export CUDA_HOME="/usr/local/cuda"

echo "CUDA Runtime Lib: $CUDA_RUNTIME_LIB"
echo "CUBLAS Lib: $CUBLAS_LIB"
echo "CUDNN Lib: $CUDNN_LIB"
echo "LD_LIBRARY_PATH set"

# Test TensorFlow GPU detection
echo ""
echo "Testing TensorFlow GPU detection..."
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPUs found:', gpus)" 2>&1 | grep -E "(GPUs found|GPU:|Could not find|Skipping)" | head -5







