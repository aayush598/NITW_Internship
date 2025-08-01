#!/bin/bash

echo "======================================="
echo "🔍 PYTHON (test_conv2d.py) OUTPUT"
echo "======================================="
python3 test_conv2d.py | sed 's/tensor(/[[/' | sed 's/)]/]]/'  # Clean up tensor format

echo
echo "======================================="
echo "🔧 VERILOG (vvp out) OUTPUT"
echo "======================================="
vvp out
