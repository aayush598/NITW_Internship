import cv2
import torch
import torch.nn as nn
import numpy as np

# Parameters (must match Verilog)
IN_CHANNELS = 2
OUT_CHANNELS = 1
IN_HEIGHT = 4
IN_WIDTH = 4
KERNEL_SIZE = 2
STRIDE = 2
PADDING = 0
BATCH_SIZE = 1
DATA_WIDTH = 8

# Output dims
OUT_HEIGHT = (IN_HEIGHT + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
OUT_WIDTH = (IN_WIDTH + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1

# Load and preprocess image
img = cv2.imread("test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IN_WIDTH, IN_HEIGHT))  # ensure size matches
img = img.astype(np.uint8)

# Use only first IN_CHANNELS channels
if IN_CHANNELS > 3:
    raise ValueError("Image has only 3 channels (RGB)")
img = img[:, :, :IN_CHANNELS]  # shape: [H, W, C]

# Convert to tensor: [B, C, H, W]
image_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
image_tensor = image_tensor.to(dtype=torch.float32)

# Flatten and save input data in NCHW order
input_flat = image_tensor.squeeze(0).to(torch.int32).numpy().reshape(-1)
np.savetxt("input_data.txt", input_flat, fmt='0x%02X')

# Create conv layer with weights=1, bias=0
conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, bias=True)
with torch.no_grad():
    conv.weight.fill_(1.0)
    conv.bias.zero_()

# Save weights (shape: [OUT_CHANNELS, IN_CHANNELS, KH, KW])
weight_flat = conv.weight.squeeze(0).reshape(-1).to(torch.int32).numpy()
np.savetxt("weights.txt", weight_flat, fmt='0x%02X')

# Save biases (should be zeros)
bias_flat = conv.bias.to(torch.int32).numpy()
np.savetxt("bias.txt", bias_flat, fmt='0x%02X')

# Run convolution and save output
with torch.no_grad():
    output = conv(image_tensor)

# Output shape: [1, OUT_CHANNELS, OUT_HEIGHT, OUT_WIDTH]
output_flat = output.squeeze(0).to(torch.int32).numpy().reshape(-1)
np.savetxt("expected_output.txt", output_flat, fmt='0x%02X')

print("✅ Input, weights, bias, and expected output saved successfully in HEX format.")
