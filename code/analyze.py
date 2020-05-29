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
from utils import WAVES

obg_dir = Path(__file__).parent.parent / "obj_dumps"
dirs = [e for e in obg_dir.iterdir() if e.is_dir()]

box_data_wave = {"control": defaultdict(list), "music": defaultdict(list)}
erd_average_control = defaultdict(list)
erd_average_music = defaultdict(list)

for i, directory in enumerate(dirs):
    (directory / "figures").mkdir(parents=True, exist_ok=True)
    files = [e for e in directory.iterdir() if e.is_file()]

    erd_mean = []

    for file in files:
        print(file)
        with open(file, 'rb') as f:
            erd_mean.append(pickle.load(f))

    for wave in WAVES:
        # Control ERD time distribution
        plt.plot([x for x in range(len(erd_mean[0][wave]))], erd_mean[0][wave])
        plt.savefig(directory / "figures" / f"{wave}_ERD_dist_control.png")
        plt.clf()
        plt.cla()
        plt.close()

        # Music time distribution
        plt.plot([x for x in range(len(erd_mean[1][wave]))], erd_mean[1][wave])
        plt.savefig(directory / "figures" / f"{wave}_ERD_dist_music.png")
        plt.clf()
        plt.cla()
        plt.close()

        # Add box plot data
        box_data_wave["control"][wave].append(erd_mean[0][wave])
        box_data_wave["music"][wave].append(erd_mean[1][wave])

        # Finding average
        erd_average_control[f"subject{i + 1}"].append((np.mean(erd_mean[0][wave])))
        erd_average_music[f"subject{i + 1}"].append(np.mean(erd_mean[1][wave]))

    erd_average_control[f"subject{i + 1}"].append(np.mean(erd_mean[0]["ABratio"]))
    erd_average_music[f"subject{i + 1}"].append(np.mean(erd_mean[1]["ABratio"]))

    # Making control ABratio time dist plots
    plt.plot([x for x in range(len(erd_mean[0]["ABratio"]))], erd_mean[0]["ABratio"])
    plt.savefig(directory / "figures" / f"ABratio_control.png")
    plt.clf()
    plt.cla()
    plt.close()

    # Making control ABratio time dist plots
    plt.plot([x for x in range(len(erd_mean[1]["ABratio"]))], erd_mean[1]["ABratio"])
    plt.savefig(directory / "figures" / f"ABratio_music.png")
    plt.clf()
    plt.cla()
    plt.close()

# Plotting box plots control
for wave in list(box_data_wave["control"]):
    ax = sns.boxplot(data=box_data_wave["control"][wave])
    ax.set(xlabel='subjects', ylabel='ERD')
    ax.figure.savefig(obg_dir / f"{wave}_boxplot_control.png")
    plt.clf()
    plt.cla()
    plt.close()
# Plotting box plots music
for wave in list(box_data_wave["music"]):
    ax = sns.boxplot(data=box_data_wave["music"][wave])
    ax.set(xlabel='subjects', ylabel='ERD')
    ax.figure.savefig(obg_dir / f"{wave}_boxplot_music.png")
    plt.clf()
    plt.cla()
    plt.close()

# Saving correlation values as xls file
control_frame = pd.DataFrame.from_dict(erd_average_control, orient="index", columns=["L1A(control)",
                                                                                     "L2A(control)",
                                                                                     "UA(control)",
                                                                                     "Theta(control)",
                                                                                     "AB_Ratio(control)"])

music_frame = pd.DataFrame.from_dict(erd_average_music, orient="index", columns=["L1A(music)",
                                                                                 "L2A(music)",
                                                                                 "UA(music)",
                                                                                 "Theta(music)",
                                                                                 "AB_Ratio(music)"])
# Normalizing values
for column in control_frame.columns:
    control_frame[column] = (control_frame[column] - control_frame[column].mean()) / (control_frame[column].std())
for column in music_frame.columns:
    music_frame[column] = (music_frame[column] - music_frame[column].mean()) / (music_frame[column].std())

writer = pd.ExcelWriter(obg_dir / "average_values.xlsx", engine='xlsxwriter')
control_frame.to_excel(writer, sheet_name="control")
music_frame.to_excel(writer, sheet_name="music")
writer.save()
writer.close()
