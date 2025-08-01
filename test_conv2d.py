import torch
import numpy as np
import torch.nn as nn
import time

# Parameters (must match Verilog)
IN_CHANNELS = 2
OUT_CHANNELS = 1
IN_HEIGHT = 4
IN_WIDTH = 4
KERNEL_SIZE = 2
STRIDE = 2
PADDING = 0
BATCH_SIZE = 1
DATA_WIDTH = 8  # bits

OUT_HEIGHT = (IN_HEIGHT + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
OUT_WIDTH = (IN_WIDTH + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
MODULO = 2 ** DATA_WIDTH  # 256 for 8-bit

def load_hex_file(filename):
    with open(filename, 'r') as f:
        hex_data = f.read().split()
    return np.array([int(x, 16) for x in hex_data], dtype=np.uint8)

# 1. Load data from hex files
input_flat = load_hex_file("input_data.txt")
weights_flat = load_hex_file("weights.txt")
bias_flat = load_hex_file("bias.txt")

# 2. Reshape input to [B, C, H, W]
input_tensor = torch.tensor(input_flat, dtype=torch.float32).reshape(
    BATCH_SIZE, IN_CHANNELS, IN_HEIGHT, IN_WIDTH
)

# 3. Reshape weights to [OUT_CHANNELS, IN_CHANNELS, KH, KW]
weight_tensor = torch.tensor(weights_flat, dtype=torch.float32).reshape(
    OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE
)

# 4. Create bias tensor
bias_tensor = torch.tensor(bias_flat, dtype=torch.float32)

# 5. Create Conv2D layer manually and assign weights/bias
conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, bias=True)
with torch.no_grad():
    conv.weight.copy_(weight_tensor)
    conv.bias.copy_(bias_tensor)

# 6. Perform convolution
with torch.no_grad():
    start_time = time.perf_counter()
    output = conv(input_tensor)
    end_time = time.perf_counter()
    elapsed_time_us = (end_time - start_time) * 1e6  # microseconds

# 7. Convert to 8-bit output using modulo
output_8bit = output.to(torch.int32) % MODULO  # simulate uint8 wrap-around

# 8. Output results
print("Output Tensor Shape:", output.shape)
print("Output Tensor (8-bit):")
print(output_8bit)
print(f"\n⏱️ Conv2D Layer Execution Time: {elapsed_time_us:.2f} µs")

# 9. Optional: Save to file in decimal
np.savetxt("expected_output.txt", output_8bit.numpy().reshape(-1), fmt='%d')

# 10. Optional: Save to file in hex
np.savetxt("expected_output_hex.txt",
           output_8bit.numpy().reshape(-1).astype(np.uint8),
           fmt='0x%02X')

