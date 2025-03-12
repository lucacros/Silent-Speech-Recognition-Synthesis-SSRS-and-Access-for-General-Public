from scipy import signal
import librosa
import numpy as np
import matplotlib.pyplot as plt

def pass_band_filter(emg_signal, low=20, high=500, fs=10000, order=4):
    """
    Pass band filter for EMG emg_signal.
    :param emg_signal: EMG emg_signal
    :param low: low frequency
    :param high: high frequency
    :param fs: sampling frequencyè
    :param order: filter order''
    """


    b, a = signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')

    # Application du filtre
    filtered_ecg = signal.filtfilt(b, a, emg_signal)

    return filtered_ecg

# audio, fs = librosa.load("DataBase/CHEEK/Bye/Bye_1.wav", sr=None, mono=False)  # 'sr=None' pour conserver la fréquence d'échantillonnage originale
# print(pass_band_filter(audio[1,:]))

def audio_to_mel_spectrogram(signal, sr=44100, n_fft=2048, hop_length=1024, n_mels=40, save_path=None):
    # Calcul du spectrogramme Mel (40,87)
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Conversion en dB
    return mel_spec_db

