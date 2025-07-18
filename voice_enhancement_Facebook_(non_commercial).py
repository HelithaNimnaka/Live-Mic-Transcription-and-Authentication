import torch
import soundfile as sf
from denoiser.pretrained import dns64

# ─── CONFIGURE THESE ─────────────────────────────────────────────────────────
# Path to your noisy WAV file:
input_wav  = "recorded_20250715_162834.wav"
# Path where you’d like your denoised output:
output_wav = "enhanced_output.wav"
# Choose your model: dns48, dns64, or master64      
model_name = "dns64"
# ─────────────────────────────────────────────────────────────────────────────

# 1) Load the model (this will download it once into your torch cache)
model = getattr(__import__("denoiser.pretrained", fromlist=[model_name]), model_name)()
model.eval()

# 2) Read your file (and mix to mono if needed)
audio, sr = sf.read(input_wav)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
# 3) Convert to a batch tensor of shape [batch, time]
batch = torch.from_numpy(audio).float().unsqueeze(0)

# 4) Denoise
with torch.no_grad():
    enhanced = model(batch)           # → shape [1, 1, time]
# squeeze out the dims → [time]
enhanced = enhanced.squeeze(0).squeeze(0).cpu().numpy()

# 5) Write back to disk
sf.write(output_wav, enhanced, sr)
print(f"Denoised audio saved to: {output_wav}")
