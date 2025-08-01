#!/bin/bash

clear

python conv2d_img.py

echo "======================================="
echo "🔍 PYTHON (test_conv2d.py) OUTPUT"
echo "======================================="
python3 test_conv2d.py | sed 's/tensor(/[[/' | sed 's/)]/]]/'  # Clean up tensor format

echo
echo "======================================="
echo "🔧 VERILOG (vvp out) OUTPUT"
echo "======================================="
iverilog -o out conv2d.v conv2d_tb.v
vvp out
