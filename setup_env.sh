#!/bin/bash
# Setup script for GPT-2 Small Implementation

# Exit on error
set -e

# Print commands
set -x

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from pyproject.toml (editable mode)
pip install -e .

# Optional: Install dev dependencies
pip install -e ".[dev]"

echo "Environment setup complete! Activate it with: source venv/bin/activate" 