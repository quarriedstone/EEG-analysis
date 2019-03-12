import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
import pickle


obg_dir = './obj_dumps/'
files = listdir(obg_dir)

erd_mean = []

for file in files:
    print(file)
    with open(obg_dir + file, 'rb') as f:
        erd_mean.append(pickle.load(f))

    #plt.plot(np.real(erd_mean["Th"]))
    #plt.show()

waves = "UA"

sns.kdeplot(np.real(erd_mean[0][waves])[100:], label="NoMusic")
sns.kdeplot(np.real(erd_mean[1][waves])[100:], label="Music")
plt.show()
sns.kdeplot(np.real(erd_mean[2][waves])[100:], label="NoMusic")
sns.kdeplot(np.real(erd_mean[3][waves])[100:], label="Music")
plt.show()