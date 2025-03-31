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

# Reads in format of int+" "+"bin_number" for bin number input, # inclusions are marked as comments, otherwise int type
def data_reader(filename: str = None):
    file = open(file=filename, mode="r")

    line = file.readline()
    line_arr = line.split(" ")
    line_arr[0] = int(line_arr[0])
    bin_number = line_arr[0]

    x_axis = []
    eof = False

    while not eof:
        try:
            inp = file.readline()
            if "#" not in inp:
                inp = int(inp)
                x_axis.append(inp)
        except:
            eof = True

    return bin_number, x_axis


bin_number, x_axis = data_reader(filename="test_data.txt")

np.random.seed(42)
fig, axs = pyplot.subplots(ncols=4, nrows=1, figsize=(12, 8))

# Original histogram with KDE overlay
n, bins, patches = axs[0].hist(x=x_axis, bins=bin_number, density=True)
axs[0].set_title('Original Histogram with KDE')
axs[0].set_ylabel('Density')

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
axs[0].plot(x_range, scaled_kde_values, 'r--', label='KDE Estimated')
axs[0].legend()

# Replotted histogram data
midpoints = (bins[:-1] + bins[1:]) / 2
histogram_data = np.column_stack((midpoints, n))
x_axis_hist = list(histogram_data[:, 0])
y_axis_hist = list(histogram_data[:, 1])

axs[1].plot(x_axis_hist, y_axis_hist, 'rx-')
axs[1].set_title('Replotted Histogram Data')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')

# Smoothed curve with peak finding (method may be inaccurate)
spline = interp.make_splrep(x_axis_hist, y_axis_hist, s=len(x_axis_hist) / 10000)
x_smooth = np.linspace(min(x_axis_hist), max(x_axis_hist), 1000)
y_smooth = interp.splev(x_smooth, spline)

peaks, _ = find_peaks(y_smooth, height=np.max(y_smooth) * 0.3, distance=20)

axs[2].plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Smoothed histogram')
axs[2].scatter(x_smooth[peaks], y_smooth[peaks], color='red', marker='^',
               label='Peaks', zorder=5)
axs[2].set_title('Smoothed Histogram with Peaks')
axs[2].set_xlabel('Value')
axs[2].set_ylabel('Frequency')
axs[2].legend()

# KDE estimation
kde_values = kde.evaluate(x_range)
axs[3].plot(x_range, kde_values, 'g.-', label='KDE Estimated')
axs[3].set_title('KDE Estimated')
axs[3].set_xlabel('Value')
axs[3].set_ylabel('Density')
axs[3].legend()

plt.tight_layout()
plt.show()