import math

import numpy as np
import scipy
from scipy.signal import savgol_filter


def move_overlaps(bss, signal, fs):
    new_bss = [bss[0]]

    for b in bss[1:]:
        lb = new_bss[-1]
        cb = b
        ocb = get_percentage_overlap(cb, lb)
        if ocb > 0:
            overlap_start = max(lb[0], cb[0])
            overlap_end = min(sum(lb[:2]), sum(cb[:2]))
            overlap_region = signal[int(overlap_start * fs):int(overlap_end * fs)]
            if len(overlap_region) == 0:
                new_bss.append(cb)
                continue
            min_val_idx = np.argmax(overlap_region)
            new_breath_sep = overlap_start + (min_val_idx / fs)
            nlbl = new_breath_sep - lb[0]
            ncbs = new_breath_sep
            ncbl = sum(cb[:2]) - new_breath_sep
            new_bss = new_bss[:-1]
            new_bss.append([lb[0], nlbl])
            new_bss.append([ncbs, ncbl])
        else:
            new_bss.append(cb)
    return new_bss


def get_percentage_overlap(b1, b2):
    l = b1[1]
    s = max(b1[0], b2[0])
    e = min(sum(b1[:2]), sum(b2[:2]))
    return max(0, (e - s) / l)


def subsume_overlaps(bss, thresh=0.8):
    new_bss = [bss[0]]
    for i in bss[1:]:
        lb = new_bss[-1]
        cb = i
        ocb = get_percentage_overlap(cb, lb)
        if ocb >= 0.99:
            continue
        olb = get_percentage_overlap(lb, cb)
        if olb >= 0.99:
            new_bss[-1] = [cb[0], cb[1]]
            continue
        if olb >= thresh or ocb >= thresh:
            start = min(lb[0], cb[0])
            end = max(sum(lb[:2]), sum(cb[:2]))
            new_bss[-1] = [start, end - start]
            continue
        new_bss.append(cb)
    return new_bss


def postprocess(bss, signal, sf):
    bss.sort(key=lambda x: x[0])
    bss = subsume_overlaps(bss)
    bss = move_overlaps(bss, signal, sf)
    return bss


def find_breaths(signal, sf):
    pdf = scipy.stats.norm(3.53, 0.79).pdf
    bss = sine_fit_bss(signal, sf, length_pdf=pdf)
    bss = postprocess(bss, signal, sf)
    return bss


def smooth(signal, win_size=51):
    return savgol_filter(signal, win_size, 3)


def de_trend(signal):
    x = np.arange(len(signal), dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    if len(y) < 2:
        return y.tolist()
    slope, intercept = np.polyfit(x, y, 1)
    return (y - (slope * x + intercept)).tolist()


def get_breath_template(duration, sf=1):
    try:
        p = 2 * math.pi / (duration * sf)
    except Exception:
        raise Exception("Breathfinder tried to divide by zero")

    return [math.sin((i * p) + 1.5 * math.pi) for i in range(duration * sf)]


def acf_unbiased(window):
    ac = np.correlate(window, window, mode="full")
    ac = ac[len(window):]
    x = []
    for i in range(len(ac)):
        x.append(ac[i] * (len(ac) / (len(ac) - i)))
    return x


def sig_corr(window, sin):
    return [
        np.corrcoef(window[i:(i + len(sin))], sin[: len(window) - i])[0, 1]
        for i in range(len(window) - 1)
    ]


def skip(i, skip_amount, overlap):
    return i, max(i + int(skip_amount * (1 - overlap)), i + 1)


def get_periodicity_candidates(window, length_pdf, sf):
    autocorrelation = acf_unbiased(window)
    peaks = []
    peaks, _ = scipy.signal.find_peaks(np.array(autocorrelation))
    if len(peaks) == 0:
        return []
    t_peaks = list(map(lambda x: x / sf, peaks))
    t_peaks = length_pdf(t_peaks)
    t_pairs = zip(peaks, t_peaks)
    return list(t_pairs)


def sine_fit_bss(
    signal,
    sampling_frequency,
    window_size=8,
    overlap=0.4,
    skip_overlap=0.95,
    correlation_threshold=0.75,
    probability_cutoff=0.0001,
    length_pdf=None,
):
    np.seterr(all="ignore")
    window_size = int(window_size * sampling_frequency)

    breaths = []
    i = 0
    last_i = -1
    signal_smooth = smooth(signal, 11)

    while i < len(signal):
        if len(signal) - i < 4 * sampling_frequency:
            break
        if i <= last_i:
            raise Exception(
                "Last position",
                str(last_i),
                "Was higher than current position",
                str(i),
                "in BSS, this may not halt",
            )
        window = signal[i:i + window_size]
        if np.std(window) == 0 or max(window) == min(window):
            last_i, i = skip(i, window_size, skip_overlap)
            continue
        window = de_trend(window)
        smoothed_window = signal_smooth[i:i + window_size]
        length_candidates = get_periodicity_candidates(window, length_pdf, sampling_frequency)
        found_breath = False
        length_candidates = list(filter(lambda x: x[1] > probability_cutoff, length_candidates))
        length_candidates.sort(key=lambda x: -x[1])
        start = None
        duration = None
        for peak, _ in length_candidates:
            breath_approximation = get_breath_template(peak, 1)
            sine_correlation = sig_corr(smoothed_window, breath_approximation)
            sc_peaks, _ = scipy.signal.find_peaks(sine_correlation)
            sc_peaks = list(sc_peaks)
            if len(sc_peaks) == 0:
                continue
            first_sample = sine_correlation[0]
            acf_increasing = sine_correlation[1] - sine_correlation[0] > 0
            if first_sample > correlation_threshold and not acf_increasing:
                sc_peaks.insert(0, 0)
            for p in sc_peaks:
                confidence = sine_correlation[p]
                if confidence > correlation_threshold:
                    found_breath = True
                    start = p
                    duration = peak
                    break
            if found_breath:
                break
        if not found_breath:
            last_i, i = skip(i, window_size, skip_overlap)
            continue

        end = start + duration
        if end >= len(window):
            last_i, i = skip(i, start, skip_overlap)
            continue
        if duration <= 0:
            last_i, i = skip(i, window_size, skip_overlap)
            continue
        breaths.append(
            [
                (i + start) / sampling_frequency,
                duration / sampling_frequency,
                sine_correlation[start],
            ]
        )

        last_i, i = skip(i, start + duration, overlap)

        if i <= last_i:
            raise Exception("Index has not progressed!")

    return breaths


def estimate_run_time(signal, sf):
    d = len(signal) / sf
    return 5.83 * 0.003 * d + 1.83


if __name__ == "__main__":
    print(
        """Welcome to the BreathFinder library.
    this library implements the breath synchronous segmentation algorithm introduced in
    my master's thesis. Running this file itself will not do much."""
    )
