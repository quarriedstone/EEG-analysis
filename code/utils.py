# Copyright (c) 2020 Herman Tarasau
# # Preprocessing
# coding: utf-8


import yaml
import numpy as np


# Function to calculate IAF
from mne.io import RawArray


def IAF(age):
    return 11.95 - 0.053 * age


# Implementing function to calculate ERD
def ERD(f, cal):
    return 100 * (cal - f) / cal


electrodes = {"central": ['EEG F3-Cz', 'EEG F4-Cz', 'EEG Fz-Cz', 'EEG C3-Cz', 'EEG C4-Cz', 'EEG P3-Cz', 'EEG P4-Cz',
                          'EEG Pz-Cz'],
              "frontal": ['EEG F7-Cz', 'EEG F3-Cz', 'EEG F4-Cz', 'EEG F8-Cz'],
              "temporal": ['EEG T3-Cz', 'EEG T4-Cz', 'EEG T5-Cz', 'EEG T6-Cz']}

CONFIG = yaml.load(open("subjects.yaml", "r"))

WAVES = ["L1A", "L2A", "UA", "Th"]


def remove_outliers(erd):
    quant25 = np.quantile(erd, 0.25)
    quant75 = np.quantile(erd, 0.75)
    erd[(erd < quant25) | (erd > quant75)] = np.median(erd)
    return erd


def calculate_arousal(alpha: RawArray, beta: RawArray):
    arousal = np.sum(beta.get_data(), axis=0) / np.sum(alpha.get_data(), axis=0)
    # valence = aF4 - aF3
    valence = alpha.get_data()[2] - alpha.get_data()[1]

    return arousal, valence
