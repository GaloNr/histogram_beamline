import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate some sample data - let's say these are measurements on an x-axis
data = np.array([2.1, 2.3, 2.2, 2.7, 3.0, 3.1, 2.5, 2.8, 2.9, 3.2, 3.5, 3.7, 3.6, 4.0, 4.2, 4.1])

# Create the kernel density estimator
kde = gaussian_kde(data)

# Create a range of x values to evaluate the KDE
x_range = np.linspace(min(data) - 0.5, max(data) + 0.5, 1000)

# Evaluate the KDE at those points
kde_values = kde(x_range)

# Plot the original data points, histogram, and the KDE
plt.figure(figsize=(10, 6))

# Plot KDE
plt.plot(x_range, kde_values, 'r-', label='KDE', linewidth=2)

# Plot histogram - normalize it to be on same scale as KDE
hist_counts, hist_bins, _ = plt.hist(data, bins=8, alpha=0.4, density=True, color='blue', label='Histogram')

# Plot the original data points
plt.scatter(data, np.zeros_like(data), c='black', marker='|', s=100, label='Data points')

plt.title('Kernel Density Estimation vs Histogram')
plt.xlabel('X value')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()