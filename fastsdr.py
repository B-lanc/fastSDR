import numpy as np


def fastsdr(original_source, predicted_source, window=88200, hop=66150):
    """
    Calculate the SDR values of a predicted source
    :params original_source: numpy array of the original source signal (shape = n_sources, n_samples, n_channels)
    :params predicted_source: numpy array of the predicted source signal (shape = n_sources, n_samples, n_channels)
    :params window: size of the SDR window
    :params hop: amount of hop for the SDR window
    :return: the SDR values
    """
    if original_source.shape != predicted_source.shape:
        raise ValueError("The shape of the sources must be the same")

    SDR = list()
    n_win = (original_source.shape[1] - window + hop) // hop
    nanmask = np.full((n_win,), False)
    for source_idx, _ in enumerate(original_source):
        source_sdr = np.arange(n_win, dtype="float64")
        for _i, _ in enumerate(source_sdr):
            orig_window = original_source[source_idx, _i * hop : _i * hop + window, :]
            pred_window = predicted_source[source_idx, _i * hop : _i * hop + window, :]

            source_sdr[_i] = _calc_sdr(orig_window, pred_window)
        nanmask = nanmask | np.isnan(source_sdr)
        SDR.append(source_sdr)
    SDR = np.stack(SDR, axis=0)
    SDR[:, nanmask] = np.nan

    return SDR


def _calc_sdr(orig_win, pred_win):
    if not orig_win.any():
        return np.nan
    return 10 * np.log10(np.sum(orig_win**2) / np.sum((orig_win - pred_win) ** 2))
