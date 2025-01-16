import matplotlib.pyplot as plt
import librosa 
from audio_analysis import read_audio, calculate_fft, calculate_spectrogram
import numpy as np

audio_path = "C:\\Users\\amirh\\OneDrive\\Documents\\signal\\Audio_process\\audio_2.ogg"
data, sr = librosa.load(audio_path, sr=None)

# Plot Waveform
time = np.linspace(0, len(data) / sr, num=len(data))
plt.figure(figsize=(12, 4))
plt.plot(time, data, alpha=0.8,color= 'green')
plt.title("Waveform in Time Domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Plot FFT
freq, fft_values = calculate_fft(data, sr)
plt.figure(figsize=(12, 4))
plt.plot(freq, fft_values, color= 'green')
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.tight_layout()
plt.show()

# Plot Spectrogram
frequencies, times, Sxx = calculate_spectrogram(data, sr)
plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='plasma')
plt.title("Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.show()
