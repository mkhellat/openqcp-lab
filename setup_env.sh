#!/bin/sh
# Setup script for openqcp-lab environment
# This script creates a virtual environment and installs dependencies.

set -e

echo "Setting up openqcp-lab Python environment..."

# Check if Python 3 is available
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is not installed or not in PATH" >&2
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Upgrade pip and install dependencies using the venv's pip
echo "Installing dependencies from requirements.txt..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo ""
echo "Setup complete. To activate the environment, run:"
echo "  . venv/bin/activate"
echo ""
echo "Then start Jupyter with, for example:"
echo "  jupyter notebook"
echo "  # or"
echo "  jupyter lab"