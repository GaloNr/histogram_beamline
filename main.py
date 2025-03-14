import matplotlib.pyplot as plt
import scipy
import numpy as np
import matplotlib

from matplotlib import pyplot
from scipy import stats
from scipy import interpolate as interp

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

fig, axs = pyplot.subplots(ncols=3, nrows=1, figsize=(12, 8))

n, bins, patches = axs[0].hist(x=x_axis, bins=bin_number)
axs[0].set_title('Original Histogram')
axs[0].set_ylabel('Frequency')

midpoints = (bins[:-1] + bins[1:]) / 2

# Create a 2D array of the histogram data
histogram_data = np.column_stack((midpoints, n))
x_axis = list(histogram_data[:, 0])
y_axis = list(histogram_data[:, 1])
print(x_axis, "\n", y_axis)

axs[1].plot(x_axis, y_axis, 'rx-')
axs[1].set_title('Replotted Histogram Data')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')

# Method 2: Creating a KDE-like smooth curve from the histogram points themselves
# using interpolation (useful when you only have the histogram points)
spline = interp.make_splrep(x_axis, y_axis, s=len(x_axis))  # s controls smoothness
x_smooth = np.linspace(min(x_axis), max(x_axis), 1000)
y_smooth = interp.splev(x_smooth, spline)

# Plot the smoothed curve
axs[2].plot(x_smooth, y_smooth, 'bx-', linewidth=2, label='Smoothed histogram')
axs[2].set_title('Replotted Histogram Data Smoothed')
axs[2].set_xlabel('Value')
axs[2].set_ylabel('Frequency')

plt.show()

#TODO Superpose smoothed graph on histogram
