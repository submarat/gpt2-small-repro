#!/bin/bash
# Script to run tests with the virtual environment

# Exit on error
set -e

# Activate virtual environment
source venv/bin/activate

# Run the tests
python tests/test_model.py

# Deactivate virtual environment
deactivate 