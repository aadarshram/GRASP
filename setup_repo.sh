#!/bin/bash

# 1. Disable the colab base env if active
if [[ "$CONDA_DEFAULT_ENV" == "colab" ]]; then
    echo "Deactivating 'colab' conda environment..."
    conda deactivate
fi

# 2. Set MUJOCO_GL environment variable
export MUJOCO_GL=egl
echo "Set MUJOCO_GL=egl"

# 3. Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first (e.g., 'curl -LsSf https://astral.sh/uv/install.sh | sh')"
    return 1 2>/dev/null || exit 1
fi

# 4. Create and activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# 5. Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt
uv pip install -e .

echo "Setup complete!"
