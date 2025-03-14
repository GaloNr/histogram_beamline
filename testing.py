import matplotlib.pyplot as plt
import scipy
import numpy as np
import matplotlib
import scipy.interpolate as interp

from matplotlib import pyplot
from scipy import stats

np.random.seed(42)
sample_data = np.concatenate([
    np.random.normal(loc=5, scale=1, size=200),
    np.random.normal(loc=10, scale=2, size=300)
])

# Extract histogram data
counts, bin_edges = np.histogram(sample_data, bins=20)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# These are our x and y values from the histogram
x_values = bin_centers
y_values = counts
print(x_values, y_values)

# Create a figure with 1 subplot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the histogram data as a regular line plot (connecting bin heights)
ax.plot(x_values, y_values, 'b-', linewidth=2, label='Histogram Line')
ax.scatter(x_values, y_values, color='blue', s=30, alpha=0.7)

# Method 2: Creating a KDE-like smooth curve from the histogram points themselves
# using interpolation (useful when you only have the histogram points)
spline = interp.make_splrep(x_values, y_values, s=3)  # s controls smoothness
x_smooth = np.linspace(min(x_values), max(x_values), 1000)
y_smooth = interp.splev(x_smooth, spline)

# Plot the smoothed curve
ax.plot(x_smooth, y_smooth, 'g--', linewidth=2, label='Smoothed histogram')

# Add labels and legend
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Line Plot and KDE from Histogram Data', fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Add text explaining the methods
ax.text(0.02, 0.97,
        "Blue line: Raw histogram data points connected\n"
        "Red line: KDE created from original sample data\n"
        "Green dashed: Spline interpolation of histogram points",
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

