# USED FOR PURELY MAIN TESTING PURPOSES

from scipy.datasets import electrocardiogram
import matplotlib.pyplot as plt
import scipy
import numpy as np
import matplotlib
import numpy.random
from matplotlib import pyplot
from scipy import stats
from scipy import interpolate as interp
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

ecg = electrocardiogram()
file = open("ecg_dataset.txt", "w")
start = 1000
end = 10000
file.write("-1" + "\n")
for i in range(start, end):
    file.write(str(ecg[i]) + "\n")
file.close()

