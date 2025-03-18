import matplotlib.pyplot as plt
import scipy
import numpy as np
import matplotlib
import numpy.random

from matplotlib import pyplot
from scipy import stats
from scipy import interpolate as interp
from scipy.stats import gaussian_kde

import kde_loocv  # requires 1d array and bw_factors 1d array of nplogspace preferred

def data_reader(filename: str = None):  # Returns data for a histogram graph with bin number
    file = open(file=filename, mode="r")

    line = file.readline()  # Reads bin number
    line_arr = line.split(" ")
    line_arr[0] = int(line_arr[0])
    bin_number = line_arr[0]

    x_axis = []  # Init of array to store values

    eof = False

    while not eof:  # File read
        try:
            inp = file.readline()
            if "#" not in inp:
                inp = int(inp)
                x_axis.append(inp)
        except:  # noqa
            eof = True

    return bin_number, x_axis


bin_number, x_axis = data_reader(filename="test_data.txt")

np.random.seed(42)
# x_axis = np.random.normal(0, 1, 1000)
# bin_number = 50

fig, axs = pyplot.subplots(ncols=4, nrows=1, figsize=(12, 8))

n, bins, patches = axs[0].hist(x=x_axis, bins=bin_number)
axs[0].set_title('Original Histogram')
axs[0].set_ylabel('Frequency')

midpoints = (bins[:-1] + bins[1:]) / 2

# Create a 2D array of the histogram data
histogram_data = np.column_stack((midpoints, n))
x_axis_hist = list(histogram_data[:, 0])
y_axis_hist = list(histogram_data[:, 1])


axs[1].plot(x_axis_hist, y_axis_hist, 'rx-')
axs[1].set_title('Replotted Histogram Data')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')

# Method 2: Creating a KDE-like smooth curve from the histogram points themselves
# using interpolation (useful when you only have the histogram points)
spline = interp.make_splrep(x_axis_hist, y_axis_hist, s=len(x_axis_hist))  # s controls smoothness
x_smooth = np.linspace(min(x_axis_hist), max(x_axis_hist), 1000)
y_smooth = interp.splev(x_smooth, spline)

# Plot the smoothed curve
axs[2].plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Smoothed histogram')
axs[2].set_title('Replotted Histogram Data Smoothed')
axs[2].set_xlabel('Value')
axs[2].set_ylabel('Frequency')

print(x_axis)
print(x_axis_hist)
kde = gaussian_kde(dataset=x_axis, bw_method='silverman')


x_range = np.linspace(min(x_axis), max(x_axis), 1000)


kde_values = kde(x_range)


axs[3].plot(x_range, kde_values, 'g.-', label='KDE Estimated')
axs[3].set_title('KDE Estimated')
axs[3].set_xlabel('Value')
axs[3].set_ylabel('Frequency')

bin_width = (x_axis_hist[1] - x_axis_hist[0])
hist_area = sum(y_axis_hist) * bin_width
scaled_kde_values = kde(x_range) * hist_area
axs[0].plot(x_range, scaled_kde_values, 'g.-', label='KDE Estimated')


plt.show()

#TODO Superpose smoothed graph on histogram
#TODO Add Peak Finding Method
#TODO Calibrate x_axis smooth point number (optional)