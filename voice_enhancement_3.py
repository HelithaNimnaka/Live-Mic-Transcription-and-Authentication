import subprocess
import os

def denoise_with_cli(input_wav, output_dir=".", model="DeepFilterNet3"):
    """
    Runs the deepFilter CLI on input_wav, placing the enhanced file in output_dir.
    model can be "DeepFilterNet", "DeepFilterNet2", or "DeepFilterNet3" (default).
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "deepFilter",
        input_wav,
        "--output-dir", output_dir,
        "--model", model,
    ]
    subprocess.run(cmd, check=True)
    enhanced_path = os.path.join(output_dir, os.path.basename(input_wav))
    return enhanced_path

# Usage
clean_file = denoise_with_cli("recorded_20250715_160338.wav")
print("Enhanced file:", clean_file)
