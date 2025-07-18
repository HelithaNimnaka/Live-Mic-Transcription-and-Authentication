import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import TclError
import wave
import sys
from speechbrain.utils.fetching import LocalStrategy
import datetime
from collections import deque
from voice_enhancement import voice_enhancement
samplerate = 44100
blocksize  = 1024
channels   = 1

# A short fixed-length deque to feed the plots
plot_buffer = deque(maxlen=blocksize)

# Open a timestamped WAV for writing
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename  = f"recorded_{timestamp}.wav"
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(2)          # 16-bit
wf.setframerate(samplerate)

# Matplotlib setup
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6))
plt.subplots_adjust(hspace=0.4)

# Energy plot
energy_vals = [0]*100
energy_line, = ax1.plot(energy_vals)
ax1.set_ylim(0,1000)
ax1.set_title("Real-Time Energy (RMS ×10 000)")

# FFT plot
freqs    = np.fft.rfftfreq(blocksize, 1/samplerate)
fft_line, = ax2.plot(freqs, np.zeros_like(freqs))
ax2.set_ylim(0,500)
ax2.set_title("Real-Time Frequency Spectrum")

def compute_energy(block):
    return np.sqrt(np.mean(block**2)) * 10000

def compute_fft(block):
    windowed = block * np.hanning(len(block))
    return np.abs(np.fft.rfft(windowed))

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    mono = indata[:,0]
    # write raw PCM to WAV
    pcm = np.int16(mono * 32767)
    # write raw PCM to WAV, ignore if file is already closed
    try:
        wf.writeframes(pcm.tobytes())
    except AttributeError:
        pass
    # enqueue for plotting
    plot_buffer.extend(mono)

def start():
    print("Recording… press Ctrl+C to stop.")
    with sd.InputStream(channels=1,
                        samplerate=samplerate,
                        blocksize=blocksize,
                        dtype='float32',
                        callback=audio_callback):
        try:
            while True:
                if len(plot_buffer) >= blocksize:
                    # grab exactly one block for plotting
                    block = np.array([plot_buffer.popleft() for _ in range(blocksize)])
                    
                    # update energy
                    e = compute_energy(block)
                    energy_vals.append(e)
                    energy_vals.pop(0)
                    energy_line.set_ydata(energy_vals)

                    # update FFT
                    fft_vals = compute_fft(block)
                    fft_line.set_ydata(fft_vals)
                    ax2.set_ylim(0, np.max(fft_vals)+10)

                    # update plots, ignore Tkinter errors
                    try:
                        fig.canvas.draw()
                        plt.pause(0.001)
                    except TclError:
                        pass
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            wf.close()
            print(f"Audio saved as '{filename}'")
            voice_enhancement(filename)
            print("Audio enhancement completed.")

if __name__ == "__main__":
    start()
