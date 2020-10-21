# Copyright (c) 2020 Herman Tarasau
# # Pre-processing
# coding: utf-8


import mne
import numpy as np
from scipy.signal import stft
import pickle
from pathlib import Path

from ..utils import IAF, ERD, CONFIG, electrodes, WAVES, remove_outliers

load_dir = Path(__file__).parent.parent.parent / "data"
obg_dir = Path(__file__).parent.parent.parent / "obj_dumps"

for subject in list(CONFIG["subjects"]):
    (obg_dir / subject).mkdir(parents=True, exist_ok=True)
    for exp_type in CONFIG["experiments"]:
        curr_dir = load_dir / subject / exp_type
        cal_name = next(curr_dir.glob("*EyesClosed.edf"))
        exp_name = next(curr_dir.glob("*MainPart.edf"))

        print(cal_name)
        raw_cal = mne.io.read_raw_edf(cal_name, preload=True)
        raw_exp = mne.io.read_raw_edf(exp_name, preload=True)
        sfreq = raw_exp.info['sfreq']

        # Data from specific channels
        eyes = raw_cal.copy().pick_channels(ch_names=electrodes["central"])
        experiment = raw_exp.copy().pick_channels(ch_names=electrodes["central"])

        # Filtering AC line noise with notch filter

        eyes_filtered_data = mne.filter.notch_filter(x=eyes.get_data(), Fs=sfreq, freqs=[50, 100])
        experiment_filtered_data = mne.filter.notch_filter(x=experiment.get_data(), Fs=sfreq, freqs=[50, 100])
        # eyes_filtered_data = eyes.get_data()
        # experiment_filtered_data = experiment.get_data()

        # Preparing data for plotting
        eyes_filtered = mne.io.RawArray(data=eyes_filtered_data,
                                        info=mne.create_info(ch_names=electrodes["central"], sfreq=sfreq))
        experiment_filtered = mne.io.RawArray(data=experiment_filtered_data,
                                              info=mne.create_info(ch_names=electrodes["central"], sfreq=sfreq))

        IAF_p = IAF(CONFIG["subjects"][subject])

        # Getting L1A, L2A, UA, Theta waves from eyes closed using FIR filtering. Also we take mean signal from all
        # channels
        eyes_sub_bands = {
            'L1A': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 4,
                                          h_freq=IAF_p - 2, sfreq=sfreq, method="fir"),
            'L2A': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 2,
                                          h_freq=IAF_p, sfreq=sfreq, method="fir"),
            'UA': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p,
                                         h_freq=IAF_p + 2, sfreq=sfreq, method="fir"),
            'Th': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 6,
                                         h_freq=IAF_p - 4, sfreq=sfreq, method="fir"),
            'Beta': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0),
                                           l_freq=IAF_p + 2,
                                           h_freq=30, sfreq=sfreq, method="fir")}

        # Getting L1A, L2A, UA, Theta waves from experiment data using FIR filtering. Also we take mean signal from all
        # channels
        experiment_sub_bands = {'L1A': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                              l_freq=IAF_p - 4, h_freq=IAF_p - 2, sfreq=sfreq,
                                                              method="fir"),
                                'L2A': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                              l_freq=IAF_p - 2, h_freq=IAF_p, sfreq=sfreq,
                                                              method="fir"),
                                'UA': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                             l_freq=IAF_p,
                                                             h_freq=IAF_p + 2, sfreq=sfreq, method="fir"),
                                'Th': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                             l_freq=IAF_p - 6, h_freq=IAF_p - 4, sfreq=sfreq,
                                                             method="fir"),
                                'Beta': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                               l_freq=IAF_p + 2, h_freq=30, sfreq=sfreq,
                                                               method="fir")}

        # Calculating calibration values. Consider mean value of all channels. Va;ue are given in microvolts
        calibration_values = {}

        for band in WAVES:
            calibration_values[band] = np.mean(eyes_sub_bands[band], axis=0) * np.power(10, 6)

        # Performing STFT transform on experiment data for each sub-band. Window size is given in samples
        window = sfreq * 2
        fft = {}

        for band in WAVES:
            fft[band] = stft(x=experiment_sub_bands[band], fs=sfreq, window=('kaiser', window), nperseg=1000)

        erd = np.vectorize(ERD)
        # Calculating ERD for experiment
        erd_mean = {}

        for band in fft:
            curr_erd = erd(fft[band][2], calibration_values[band])
            erd_mean[band] = remove_outliers(np.real(np.mean(curr_erd, axis=0)))

        # Adding clean Beta and UA energy ratio
        erd_mean["ABratio"] = remove_outliers(np.real(np.power(experiment_sub_bands["UA"] / experiment_sub_bands["Beta"], 2)))

        # Dumping erd_mean of experiment
        pickle.dump(erd_mean, open(obg_dir / subject / "".join([subject, exp_type, ".pkl"]), 'wb'))
