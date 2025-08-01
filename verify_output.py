import numpy as np

# === File Paths ===
VERILOG_OUTPUT_PATH = "verilog_output.txt"
PYTHON_OUTPUT_PATH = "expected_output.txt"

def load_output_file(path: str) -> np.ndarray:
    """Loads a text file of integers (one per line) into a NumPy array."""
    try:
        return np.loadtxt(path, dtype=int)
    except Exception as e:
        raise RuntimeError(f"Error reading file '{path}': {e}")

def compare_outputs(verilog_out: np.ndarray, python_out: np.ndarray):
    """Compares two arrays and reports differences."""
    if verilog_out.shape != python_out.shape:
        raise ValueError(f"Shape mismatch: Verilog {verilog_out.shape}, Python {python_out.shape}")

    mismatches = np.where(verilog_out != python_out)[0]
    total = len(verilog_out)
    errors = len(mismatches)

    if errors == 0:
        print(f"✅ PASS: All {total} output values match.")
    else:
        print(f"❌ FAIL: {errors}/{total} outputs mismatched.")
        print("\nIndex | Verilog | Python")
        print("----------------------------")
        for i in mismatches:
            print(f"{i:5d} | {verilog_out[i]:7d} | {python_out[i]:6d}")

def main():
    verilog_output = load_output_file(VERILOG_OUTPUT_PATH)
    python_output = load_output_file(PYTHON_OUTPUT_PATH)
    compare_outputs(verilog_output, python_output)

if __name__ == "__main__":
    main()
