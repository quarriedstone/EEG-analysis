# Copyright (c) 2020 Herman Tarasau
# # Pre-processing
# coding: utf-8


import mne
import numpy as np
from scipy.signal import stft
import pickle
from pathlib import Path

from code.utils import IAF, ERD, CONFIG, electrodes, WAVES, remove_outliers, calculate_arousal, calculate_valence

load_dir = Path(__file__).parent.parent.parent / "data"
obg_dir = Path(__file__).parent.parent.parent / "obj_dumps"

for subject in list(CONFIG["subjects"]):
    (obg_dir / subject).mkdir(parents=True, exist_ok=True)
    for exp_type in CONFIG["experiments"]:
        curr_dir = load_dir / subject / exp_type
        print(curr_dir)
        exp_name = next(curr_dir.glob("*MainPart.edf"))

        raw_exp = mne.io.read_raw_edf(exp_name, preload=True)
        sfreq = raw_exp.info['sfreq']

        # Data from specific channels
        experiment = raw_exp.copy().pick_channels(ch_names=electrodes["frontal"])

        # Filtering AC line noise with notch filter
        experiment_filtered_data = mne.filter.notch_filter(x=experiment.get_data(), Fs=sfreq, freqs=[50, 100])
        # eyes_filtered_data = eyes.get_data()
        # experiment_filtered_data = experiment.get_data()

        # Preparing data for plotting
        experiment_filtered = mne.io.RawArray(data=experiment_filtered_data,
                                              info=mne.create_info(ch_names=electrodes["frontal"], sfreq=sfreq))

        # Getting L1A, L2A, UA, Theta waves from experiment data using FIR filtering. Also we take mean signal from all
        # channels
        experiment_sub_bands = {
            'Alpha': mne.io.RawArray(
                mne.filter.filter_data(data=experiment_filtered.get_data(),
                                       l_freq=8,
                                       h_freq=12,
                                       sfreq=sfreq,
                                       method="fir"),
                info=mne.create_info(ch_names=electrodes["frontal"],
                                     sfreq=sfreq)),
            'Beta': mne.io.RawArray(
                mne.filter.filter_data(data=experiment_filtered.get_data(),
                                       l_freq=12,
                                       h_freq=28,
                                       sfreq=sfreq,
                                       method="fir"),
                info=mne.create_info(ch_names=electrodes["frontal"],
                                     sfreq=sfreq))
        }
        arousal_data = remove_outliers(calculate_arousal(alpha=experiment_sub_bands["Alpha"],
                                                         beta=experiment_sub_bands["Beta"]))
        valence_data = remove_outliers(calculate_valence(alpha=experiment_sub_bands["Alpha"]))

        # Dumping arousal and  of experiment
        pickle.dump({"arousal": arousal_data,
                     "valence": valence_data},
                    open(obg_dir / subject / "".join([subject, exp_type, ".pkl"]), 'wb'))
