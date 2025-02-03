from scipy import signal
import librosa

def pass_band_filter(emg_signal, low=20, high=500, fs=44100, order=4):
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
