import pickle
import time

import librosa
import numpy as np
from scipy.signal import stft, istft

import museval
from fastsdr import fastsdr
from museparation.scripts.get_musdb import get_musdbhq


def main(path):
    """
    saves a pickle file, the format of the pickle file is
    {
        song0 : {
            SDR : nparray SDR values from bss_eval,
            fastsdr : nparray SDR values from fastsdr,
            SDRtime : time to calculate the SDR from bss_eval,
            fastsdrtime : time to calculate the SDR from fastsdr
        },
        song1 : {

        },
        ...
    }
    """
    musdb = get_musdbhq(path)
    eps = np.finfo(float).eps

    results = dict()
    for song in musdb["test"]:
        song_name = song["mixture"].split("/")[-2]
        results[song_name] = dict()

        bass, _ = librosa.load(song["bass"], sr=44100, mono=False)
        drums, _ = librosa.load(song["drums"], sr=44100, mono=False)
        other, _ = librosa.load(song["other"], sr=44100, mono=False)
        vocals, _ = librosa.load(song["vocals"], sr=44100, mono=False)
        mixture, _ = librosa.load(song["mixture"], sr=44100, mono=False)

        bass_stft = stft(bass, nperseg=256, noverlap=128, boundary=None)[-1]
        drums_stft = stft(drums, nperseg=256, noverlap=128, boundary=None)[-1]
        other_stft = stft(other, nperseg=256, noverlap=128, boundary=None)[-1]
        vocals_stft = stft(vocals, nperseg=256, noverlap=128, boundary=None)[-1]
        mixture_stft = stft(mixture, nperseg=256, noverlap=128, boundary=None)[-1]

        mixture_abs = np.abs(mixture_stft) + eps
        bass_IRM = np.abs(bass_stft) / mixture_abs
        drums_IRM = np.abs(drums_stft) / mixture_abs
        other_IRM = np.abs(other_stft) / mixture_abs
        vocals_IRM = np.abs(vocals_stft) / mixture_abs

        bass_stft2 = mixture_stft * bass_IRM
        drums_stft2 = mixture_stft * drums_IRM
        other_stft2 = mixture_stft * other_IRM
        vocals_stft2 = mixture_stft * vocals_IRM

        bass_preds = istft(bass_stft2, nperseg=256, noverlap=128, boundary=False)[-1]
        drums_preds = istft(drums_stft2, nperseg=256, noverlap=128, boundary=False)[-1]
        other_preds = istft(other_stft2, nperseg=256, noverlap=128, boundary=False)[-1]
        vocals_preds = istft(vocals_stft2, nperseg=256, noverlap=128, boundary=False)[
            -1
        ]

        if bass.shape != bass_preds.shape:
            bass_preds = bass_preds[:, : bass.shape[1]]
            drums_preds = drums_preds[:, : drums.shape[1]]
            other_preds = other_preds[:, : other.shape[1]]
            vocals_preds = vocals_preds[:, : vocals.shape[1]]

        original_source = np.stack((bass.T, drums.T, other.T, vocals.T))
        predicted_source = np.stack(
            (bass_preds.T, drums_preds.T, other_preds.T, vocals_preds.T)
        )

        t1 = time.perf_counter()
        SDR, _, _, _, _ = museval.metrics.bss_eval(original_source, predicted_source)
        t2 = time.perf_counter()
        results[song_name]["SDR"] = SDR
        results[song_name]["SDRtime"] = t2 - t1
        print(song_name, t2 - t1)

        t1 = time.perf_counter()
        SDR = fastsdr(original_source, predicted_source)
        t2 = time.perf_counter()
        results[song_name]["fastsdr"] = SDR
        results[song_name]["fastsdrtime"] = t2 - t1
        print(song_name, t2 - t1)

    with open("results2.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # if you wanna try for yourself
    ##############################################
    PATH = "/mnt/Data/MachineLearning/Datasets/musdb18hq"
    ##############################################
    main(PATH)
