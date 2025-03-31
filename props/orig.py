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
fig, axs = pyplot.subplots(figsize=(7, 9))

# Original histogram with KDE overlay
n, bins, patches = axs.hist(x=x_axis, bins=bin_number, density=True)
axs.set_title('Original Histogram')
axs.set_ylabel('Density')

plt.savefig(fname=r"C:\Users\Comp\Desktop\beamline_props\ecgorig1000-10000.png")
