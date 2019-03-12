# coding: utf-8

# # Preprocessing

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import seaborn as sns
from os import listdir
import glob
import pickle


# Function to calculate IAF
def IAF(age):
    return 11.95 - 0.053 * age


# Implementing function to calculate ERD
def ERD(f, cal):
    return 100 * (cal - f) / cal


electrodes_list = ['EEG F3-Cz', 'EEG F4-Cz', 'EEG Fz-Cz', 'EEG C3-Cz', 'EEG C4-Cz', 'EEG P3-Cz', 'EEG P4-Cz',
                   'EEG Pz-Cz']

load_dir = './EEG_Export/'
obg_dir = './obj_dumps/'
directories = listdir(load_dir)

for directory in directories:
    cal_name = glob.glob(load_dir + directory + "/*Calibration.edf")
    exp_name = glob.glob(load_dir + directory + "/*.Data.edf")

    print(cal_name)
    raw_cal = mne.io.read_raw_edf(cal_name[0], preload=True)
    raw_exp = mne.io.read_raw_edf(exp_name[0], preload=True)
    sfreq = raw_exp.info['sfreq']

    # Data from specific channels
    eyes = raw_cal.copy().pick_channels(ch_names=electrodes_list);
    experiment = raw_exp.copy().pick_channels(ch_names=electrodes_list)

    # Filtering AC line noise with notch filter

    # eyes_filtered_data = mne.filter.notch_filter(x=eyes.get_data(), Fs=sfreq, freqs=[50, 100])
    # experiment_filtered_data = mne.filter.notch_filter(x=experiment.get_data(), Fs=sfreq, freqs=[50, 100])
    eyes_filtered_data = eyes.get_data()
    experiment_filtered_data = experiment.get_data()

    # Preparing data for plotting
    eyes_filtered = mne.io.RawArray(data=eyes_filtered_data,
                                    info=mne.create_info(ch_names=electrodes_list, sfreq=sfreq))
    experiment_filtered = mne.io.RawArray(data=experiment_filtered_data,
                                          info=mne.create_info(ch_names=electrodes_list, sfreq=sfreq))

    IAF_p = IAF(20)

    # Getting L1A, L2A, UA, Theta waves from eyes closed using FIR filtering. Also we take mean signal from all channels
    eyes_sub_bands = {}

    eyes_sub_bands['L1A'] = mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 4,
                                                   h_freq=IAF_p - 2, sfreq=sfreq, method="fir")
    eyes_sub_bands['L2A'] = mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 2,
                                                   h_freq=IAF_p, sfreq=sfreq, method="fir")
    eyes_sub_bands['UA'] = mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p,
                                                  h_freq=IAF_p + 2, sfreq=sfreq, method="fir")
    eyes_sub_bands['Th'] = mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 6,
                                                  h_freq=IAF_p - 4, sfreq=sfreq, method="fir")

    # Getting L1A, L2A, UA, Theta waves from experiment data using FIR filtering. Also we take mean signal from all
    # channels
    experiment_sub_bands = {}

    experiment_sub_bands['L1A'] = mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                         l_freq=IAF_p - 4, h_freq=IAF_p - 2, sfreq=sfreq, method="fir")
    experiment_sub_bands['L2A'] = mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                         l_freq=IAF_p - 2, h_freq=IAF_p, sfreq=sfreq, method="fir")
    experiment_sub_bands['UA'] = mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                        l_freq=IAF_p,
                                                        h_freq=IAF_p + 2, sfreq=sfreq, method="fir")
    experiment_sub_bands['Th'] = mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                        l_freq=IAF_p - 6, h_freq=IAF_p - 4, sfreq=sfreq, method="fir")

    # Calculating calibration values. Consider mean value of all channels. Va;ue are given in microvolts
    calibration_values = {}

    for band in eyes_sub_bands:
        calibration_values[band] = np.mean(eyes_sub_bands[band], axis=0) * np.power(10, 6)

    # Performing STFT transform on experiment data for each sub-band. Window size is given in samples
    window = sfreq * 2
    fft = {}

    for band in experiment_sub_bands:
        fft[band] = stft(x=experiment_sub_bands[band], fs=sfreq, window=('kaiser', window), nperseg=1000)

    erd = np.vectorize(ERD)
    # Calculating ERD for experiment
    erd_mean = {}
    erd_all = {}

    for band in fft:
        erd_all[band] = erd(fft[band][2], calibration_values[band])
        erd_mean[band] = np.mean(erd_all[band], axis=0)

    # Dumping erd_mean of experiment
    pickle.dump(erd_mean, open(obg_dir + directory + ".pkl", 'wb'))

    #plt.plot(np.real(erd_mean["Th"]))
    #plt.show()
    #sns.kdeplot(np.real(erd_mean["Th"])[100:])
    #plt.show()
