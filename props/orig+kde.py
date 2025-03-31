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
fig, axs = pyplot.subplots(figsize=(9, 16))

# Original histogram with KDE overlay
n, bins, patches = axs.hist(x=x_axis, bins=bin_number, density=True)
axs.set_title('Original Histogram with KDE')
axs.set_ylabel('Density')

# Calculate KDE with proper scaling
kde = gaussian_kde(dataset=x_axis, bw_method='silverman')
bin_width = bins[1] - bins[0]
hist_area = sum(n) * bin_width
x_range = np.linspace(min(x_axis), max(x_axis), 1000)

# Scale KDE to match histogram height
kde_values = kde.evaluate(x_range)
max_kde = np.max(kde_values)
max_hist = np.max(n)
scaling_factor = max_hist / max_kde
scaled_kde_values = kde_values * scaling_factor

# Overlay scaled KDE on original histogram
axs.plot(x_range, scaled_kde_values, 'r--', label='KDE Estimated')
axs.legend()

plt.savefig(fname=r"C:\Users\Comp\Desktop\beamline_props\ecgorigkde1000-10000.png")