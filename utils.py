def pass_band_filter(signal, low=50, high=500, fs=44100, order=5):
    """
    Pass band filter for EMG signal.
    :param signal: EMG signal
    :param low: low frequency
    :param high: high frequency
    :param fs: sampling frequencyè
    :param order: filter order''
    """
    b, a = signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')

    # Application du filtre
    filtered_ecg = signal.filtfilt(b, a, signal)

    return filtered_ecg