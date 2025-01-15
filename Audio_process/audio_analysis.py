# audio_processing.py
import numpy as np
from scipy.signal import spectrogram
from scipy.fftpack import fft
import soundfile as sf

def read_audio(file_path):
    """Reads an audio file and converts it to mono if necessary."""
    data, sr = sf.read(file_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return data, sr

def calculate_fft(data, sr):
    """Calculates the FFT and returns frequency and magnitude."""
    n = len(data)
    freq = np.fft.rfftfreq(n, d=1/sr)
    fft_values = np.abs(fft(data)[:len(freq)])
    return freq, fft_values

def calculate_spectrogram(data, sr):
    """Calculates the spectrogram of the signal."""
    frequencies, times, Sxx = spectrogram(data, fs=sr)
    return frequencies, times, Sxx

def save_plot(fig, filename):
    """Saves a matplotlib figure to a file."""
    fig.savefig(filename, dpi=300)
    print(f"Plot saved as {filename}")
