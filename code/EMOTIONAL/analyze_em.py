# Copyright (c) 2020 Herman Tarasau
# # Preprocessing
# coding: utf-8


import pickle
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from code.utils import ARVAL

obg_dir = Path(__file__).parent.parent.parent / "obj_dumps"
dirs = [e for e in obg_dir.iterdir() if e.is_dir()]

box_data_wave = {"control": defaultdict(list), "music": defaultdict(list)}
arval_average_control = defaultdict(list)
arval_average_music = defaultdict(list)

for i, directory in enumerate(dirs):
    (directory / "figures").mkdir(parents=True, exist_ok=True)
    files = [e for e in directory.iterdir() if e.is_file()]

    print(files)
    arval_dicts = []

    for file in files:
        print(file)
        with open(file, 'rb') as f:
            arval_dicts.append(pickle.load(f))

    for arval_type in ARVAL:
        # Control arousal distribution
        plt.plot([x for x in range(len(arval_dicts[0][arval_type]))], arval_dicts[0][arval_type])
        plt.savefig(directory / "figures" / f"{arval_type}_dist_control.png")
        plt.clf()
        plt.cla()
        plt.close()

        # Music arousal distribution
        plt.plot([x for x in range(len(arval_dicts[1][arval_type]))], arval_dicts[1][arval_type])
        plt.savefig(directory / "figures" / f"{arval_type}_dist_music.png")
        plt.clf()
        plt.cla()
        plt.close()

        # Add box plot data
        box_data_wave["control"][arval_type].append(arval_dicts[0][arval_type])
        box_data_wave["music"][arval_type].append(arval_dicts[1][arval_type])

        # Finding average
        arval_average_control[f"subject{i + 1}"].append((np.mean(arval_dicts[0][arval_type])))
        arval_average_music[f"subject{i + 1}"].append(np.mean(arval_dicts[1][arval_type]))

# Plotting box plots control
for arval_type in list(box_data_wave["control"]):
    ax = sns.boxplot(data=box_data_wave["control"][arval_type])
    ax.set(xlabel='subjects', ylabel=f'{arval_type}')
    ax.figure.savefig(obg_dir / f"{arval_type}_boxplot_control.png")
    plt.clf()
    plt.cla()
    plt.close()
# Plotting box plots music
for arval_type in list(box_data_wave["music"]):
    ax = sns.boxplot(data=box_data_wave["music"][arval_type])
    ax.set(xlabel='subjects', ylabel=f'{arval_type}')
    ax.figure.savefig(obg_dir / f"{arval_type}_boxplot_music.png")
    plt.clf()
    plt.cla()
    plt.close()

# Saving correlation values as xls file
control_frame = pd.DataFrame.from_dict(arval_average_control, orient="index", columns=["Arousal", "Valence"])

music_frame = pd.DataFrame.from_dict(arval_average_music, orient="index", columns=["Arousal", "Valence"])

# # Normalizing values
# for column in control_frame.columns:
#     control_frame[column] = (control_frame[column] - control_frame[column].mean()) / (control_frame[column].std())
# for column in music_frame.columns:
#     music_frame[column] = (music_frame[column] - music_frame[column].mean()) / (music_frame[column].std())

writer = pd.ExcelWriter(obg_dir / "average_values.xlsx", engine='xlsxwriter')
control_frame.to_excel(writer, sheet_name="control")
music_frame.to_excel(writer, sheet_name="music")
writer.save()
writer.close()
