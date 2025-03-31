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
from numpy import histogram_bin_edges

# Reads in format of float+" "+"bin_number" for bin number input, # inclusions are marked as comments, otherwise float type
def data_reader(filename: str = None):
    file = open(file=filename, mode="r")

    line = file.readline()
    line_arr = line.split(" ")
    line_arr[0] = float(line_arr[0].strip("\n"))
    bin_number = None if line_arr[0] < 0 else line_arr[0]

    x_axis = []
    eof = False

    while not eof:
        try:
            inp = file.readline()
            inp = inp.strip("\n")
            if "#" not in inp:
                inp = float(inp)
                x_axis.append(inp)
        except:
            eof = True

    return bin_number, x_axis


bin_number, x_axis = data_reader(filename=r"C:\Users\Comp\PycharmProjects\histogram_beamline\ecg_dataset.txt")
print(x_axis)
if bin_number == None:
    bin_number = len(histogram_bin_edges(x_axis, bins='scott'))
print(bin_number)

np.random.seed(42)
fig, axs = pyplot.subplots(nrows=1, ncols=2, figsize=(18, 16))

# Original histogram with KDE overlay
n, bins, patches = axs[0].hist(x=x_axis, bins=bin_number, density=True)
axs[0].set_title('Original Histogram with KDE')
axs[0].set_ylabel('Density')

# Replotted histogram data
midpoints = (bins[:-1] + bins[1:]) / 2
histogram_data = np.column_stack((midpoints, n))
x_axis_hist = list(histogram_data[:, 0])
y_axis_hist = list(histogram_data[:, 1])

axs[1].plot(x_axis_hist, y_axis_hist, 'rx-')
axs[1].set_title('Replotted Histogram Data')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')

plt.savefig(fname=r"C:\Users\Comp\Desktop\beamline_props\ecgreplotted1000-10000.png")
