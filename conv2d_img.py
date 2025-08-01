import cv2
import torch
import torch.nn as nn
import numpy as np
import os

# === Configuration === #
IN_CHANNELS = 2
OUT_CHANNELS = 1
IN_HEIGHT = 8
IN_WIDTH = 8
KERNEL_SIZE = 2
STRIDE = 2
PADDING = 0
BATCH_SIZE = 1
DATA_WIDTH = 8  # bits

OUT_HEIGHT = (IN_HEIGHT + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
OUT_WIDTH = (IN_WIDTH + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1

INPUT_IMAGE_PATH = "test.png"
OUTPUT_DIR = "."
IMAGE_OUTPUT_PATHS = {
    "input": os.path.join(OUTPUT_DIR, "input_data.txt"),
    "weights": os.path.join(OUTPUT_DIR, "weights.txt"),
    "bias": os.path.join(OUTPUT_DIR, "bias.txt"),
    "output": os.path.join(OUTPUT_DIR, "expected_output.txt")
}

# === Functions === #

def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess image to match input shape for Conv2D."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IN_WIDTH, IN_HEIGHT))
    img = img.astype(np.uint8)

    if IN_CHANNELS > 3:
        raise ValueError("Image only supports up to 3 channels (RGB)")

    img = img[:, :, :IN_CHANNELS]
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    assert tensor.shape == (1, IN_CHANNELS, IN_HEIGHT, IN_WIDTH)
    return tensor

def save_tensor_flat_hex(tensor: torch.Tensor, path: str):
    """Save a flattened tensor as hex integers."""
    flat = tensor.to(torch.int32).numpy().reshape(-1)
    np.savetxt(path, flat, fmt='%02X')

def create_conv_layer() -> nn.Conv2d:
    """Create Conv2D layer with all weights = 1 and biases = 0."""
    conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, bias=True)
    with torch.no_grad():
        conv.weight.fill_(1.0)
        conv.bias.zero_()
    return conv

def save_weights_bias(conv: nn.Conv2d, weight_path: str, bias_path: str):
    """Save flattened weights and biases to hex files."""
    weights = conv.weight.detach().reshape(-1).to(torch.int32).numpy()
    bias = conv.bias.detach().to(torch.int32).numpy()
    np.savetxt(weight_path, weights, fmt='%02X')
    np.savetxt(bias_path, bias, fmt='%02X')

def run_convolution(conv: nn.Conv2d, input_tensor: torch.Tensor) -> torch.Tensor:
    """Run convolution and return output tensor."""
    with torch.no_grad():
        return conv(input_tensor)

# === Main Flow === #

def main():
    # 1. Load and process input image
    input_tensor = preprocess_image(INPUT_IMAGE_PATH)
    
    # 2. Save input tensor in hex
    save_tensor_flat_hex(input_tensor.squeeze(0), IMAGE_OUTPUT_PATHS["input"])
    
    # 3. Create and initialize convolution layer
    conv = create_conv_layer()
    
    # 4. Save weights and biases
    save_weights_bias(conv, IMAGE_OUTPUT_PATHS["weights"], IMAGE_OUTPUT_PATHS["bias"])
    
    # 5. Run convolution
    output = run_convolution(conv, input_tensor)

    # 6. Save output tensor
    save_tensor_flat_hex(output.squeeze(0), IMAGE_OUTPUT_PATHS["output"])

    print("✅ All tensors saved successfully in HEX format.")

if __name__ == "__main__":
    main()
