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
fig, axs = pyplot.subplots(ncols=2, nrows=1, figsize=(18, 16))

n, bins, patches = axs[0].hist(x=x_axis, bins=bin_number, density=True)

# Replotted histogram data
midpoints = (bins[:-1] + bins[1:]) / 2
histogram_data = np.column_stack((midpoints, n))
x_axis_hist = list(histogram_data[:, 0])
y_axis_hist = list(histogram_data[:, 1])

# Smoothed curve with peak finding (method may be inaccurate)
spline = interp.make_splrep(x_axis_hist, y_axis_hist, s=len(x_axis_hist) / 10000)
x_smooth = np.linspace(min(x_axis_hist), max(x_axis_hist), 1000)
y_smooth = interp.splev(x_smooth, spline)

peaks, _ = find_peaks(y_smooth, height=np.max(y_smooth) * 0.3, distance=20)

axs[1].plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Smoothed histogram')
axs[1].scatter(x_smooth[peaks], y_smooth[peaks], color='red', marker='^',
               label='Peaks', zorder=5)
axs[1].set_title('Smoothed Histogram with Peaks')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')
axs[1].legend()

plt.savefig(fname=r"C:\Users\Comp\Desktop\beamline_props\ecgsmooth1000-10000.png")
