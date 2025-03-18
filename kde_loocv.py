'''import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Sample data
data = np.array([2.1, 2.3, 2.2, 2.7, 3.0, 3.1, 2.5, 2.8, 2.9, 3.2, 3.5, 3.7, 3.6, 4.0, 4.2, 4.1])

# Define a range of bandwidths to test (using scipy's bw_factor, which is a multiplier to Scott's rule)
bw_factors = np.logspace(-1, 1, 20)  # From 0.1 to 10 times Scott's rule


# Define our LOOCV function for KDE bandwidth
def kde_loocv(data, bw_factors):
    n = len(data)
    cv_scores = []

    # For each bandwidth factor
    for bw_factor in bw_factors:
        fold_scores = []

        # For each point in the dataset (leave-one-out)
        for i in range(n):
            # Training set: all points except i
            train_data = np.delete(data, i)

            # Test point
            test_point = np.array([data[i]])

            # Fit KDE on training data with current bandwidth factor
            kde = gaussian_kde(train_data, bw_method=bw_factor)

            # Evaluate log-density at the test point
            log_density = np.log(kde(test_point)[0])
            fold_scores.append(log_density)

        # Average log-density for this bandwidth
        cv_scores.append(np.mean(fold_scores))

    return cv_scores


# Run the custom LOOCV
cv_scores = kde_loocv(data, bw_factors)

# Find optimal bandwidth factor
optimal_idx = np.argmax(cv_scores)
optimal_bw_factor = bw_factors[optimal_idx]

# Get actual bandwidth value (in data units) for the optimal factor
# We need to create a KDE with this factor to see what bandwidth it corresponds to
kde_optimal = gaussian_kde(data, bw_method=optimal_bw_factor)
optimal_bandwidth = kde_optimal.factor * np.std(data) * len(data) ** (-1 / 5)

print(f"Optimal bandwidth factor: {optimal_bw_factor:.4f}")
print(f"Corresponding bandwidth value: {optimal_bandwidth:.4f}")

# Visualize the results
plt.figure(figsize=(10, 5))
plt.semilogx(bw_factors, cv_scores)
plt.axvline(optimal_bw_factor, color='r', linestyle='--',
            label=f'Optimal factor: {optimal_bw_factor:.4f}')
plt.xlabel('Bandwidth factor')
plt.ylabel('Average log-density')
plt.title('Custom Leave-One-Out Cross-Validation for KDE Bandwidth Selection')
plt.legend()
plt.grid(True)
plt.show()

# Show the resulting optimal KDE
x_range = np.linspace(min(data) - 0.5, max(data) + 0.5, 1000)
plt.figure(figsize=(10, 5))
plt.hist(data, bins=8, alpha=0.4, density=True, color='blue', label='Histogram')
plt.plot(x_range, kde_optimal(x_range), 'r-', linewidth=2,
         label=f'KDE with optimal bandwidth={optimal_bandwidth:.4f}')
plt.axhline(y=0, color='k', alpha=0.3)
plt.scatter(data, np.zeros_like(data), c='black', marker='|', s=100, label='Data points')
plt.title('Optimal KDE from Custom LOOCV')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()'''

'''import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec

# Sample data
data = np.array([2.1, 2.3, 2.2, 2.7, 3.0, 3.1, 2.5, 2.8, 2.9, 3.2, 3.5, 3.7, 3.6, 4.0, 4.2, 4.1])

# Reduce the number of bandwidth factors to test (using just 4 for better visualization)
bw_factors = np.array([0.3, 0.7, 1.5, 3.0])


# Function to create KDE with a specific bandwidth factor
def create_kde(data, bw_factor):
    return gaussian_kde(data, bw_method=bw_factor)


# Function to calculate actual bandwidth value from factor
def get_actual_bandwidth(kde):
    return kde.factor * np.std(kde.dataset[0]) * len(kde.dataset[0]) ** (-1 / 5)


# LOOCV function that also returns intermediate results for visualization
def kde_loocv_with_steps(data, bw_factors):
    n = len(data)
    cv_scores = []

    # Store all partial KDEs for visualization
    all_partial_kdes = []

    # For each bandwidth factor
    for bw_factor in bw_factors:
        fold_scores = []
        partial_kdes = []

        # For each point in the dataset (leave-one-out)
        for i in range(n):
            # Training set: all points except i
            train_data = np.delete(data, i)

            # Test point
            test_point = np.array([data[i]])

            # Fit KDE on training data with current bandwidth factor
            kde = gaussian_kde(train_data, bw_method=bw_factor)

            # Store this partial KDE for visualization
            partial_kdes.append((i, kde, test_point[0]))

            # Evaluate log-density at the test point
            log_density = np.log(kde(test_point)[0])
            fold_scores.append(log_density)

        # Average log-density for this bandwidth
        cv_scores.append(np.mean(fold_scores))
        all_partial_kdes.append((bw_factor, partial_kdes))

    return cv_scores, all_partial_kdes


# Run the custom LOOCV with step recording
cv_scores, all_partial_kdes = kde_loocv_with_steps(data, bw_factors)

# Find optimal bandwidth factor
optimal_idx = np.argmax(cv_scores)
optimal_bw_factor = bw_factors[optimal_idx]

# Create figure with multiple subplots
plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, height_ratios=[1, 2.5, 1])

# Common x-range for all plots
x_range = np.linspace(min(data) - 0.5, max(data) + 0.5, 1000)

# 1. Plot cross-validation scores
ax_scores = plt.subplot(gs[0, :])
ax_scores.plot(bw_factors, cv_scores, 'o-', color='blue')
ax_scores.axvline(optimal_bw_factor, color='r', linestyle='--',
                  label=f'Optimal factor: {optimal_bw_factor:.2f}')
ax_scores.set_xlabel('Bandwidth factor')
ax_scores.set_ylabel('Avg Log-density')
ax_scores.set_title('LOOCV Scores by Bandwidth Factor')
ax_scores.legend()
ax_scores.grid(True)

# 2. Plot all bandwidths with full data
ax_full = plt.subplot(gs[1, 0])
# First create histogram
ax_full.hist(data, bins=8, alpha=0.4, density=True, color='gray', label='Histogram')
# Plot KDEs for all bandwidth factors
for bw_factor in bw_factors:
    kde = gaussian_kde(data, bw_method=bw_factor)
    actual_bw = get_actual_bandwidth(kde)
    ax_full.plot(x_range, kde(x_range), '-',
                 label=f'BW factor={bw_factor:.2f} (Actual={actual_bw:.2f})')
ax_full.axhline(y=0, color='k', alpha=0.3)
ax_full.scatter(data, np.zeros_like(data), c='black', marker='|', s=100, label='Data points')
ax_full.set_title('Full Data KDEs with Different Bandwidths')
ax_full.legend()
ax_full.grid(True, alpha=0.3)

# 3. Plot LOOCV results
ax_optimal = plt.subplot(gs[1, 1])
kde_optimal = gaussian_kde(data, bw_method=optimal_bw_factor)
optimal_actual_bw = get_actual_bandwidth(kde_optimal)
ax_optimal.hist(data, bins=8, alpha=0.4, density=True, color='gray', label='Histogram')
ax_optimal.plot(x_range, kde_optimal(x_range), 'r-', linewidth=2,
                label=f'Optimal KDE (BW={optimal_actual_bw:.2f})')
ax_optimal.axhline(y=0, color='k', alpha=0.3)
ax_optimal.scatter(data, np.zeros_like(data), c='black', marker='|', s=100, label='Data points')
ax_optimal.set_title('Optimal KDE from LOOCV')
ax_optimal.legend()
ax_optimal.grid(True, alpha=0.3)

# 4. Show LOOCV examples for one specific bandwidth factor (use the optimal one)
# Create small multiples showing a few leave-one-out examples
ax_examples = plt.subplot(gs[2, :])

# Select the optimal bandwidth factor's leave-one-out KDEs
optimal_results = next(results for bw_factor, results in all_partial_kdes if bw_factor == optimal_bw_factor)

# Select 4 examples for demonstration (first, last, and two middle points)
example_indices = [0, len(data) // 3, 2 * len(data) // 3, len(data) - 1]

# Common y-limit for small multiples
y_max = 0

# First pass to determine y-axis limits
for idx in example_indices:
    i, kde, test_point = optimal_results[idx]
    y_vals = kde(x_range)
    y_max = max(y_max, np.max(y_vals) * 1.2)

# Create small multiple plots
for j, idx in enumerate(example_indices):
    i, kde, test_point = optimal_results[idx]

    # Define subplot position (left, bottom, width, height)
    left = 0.05 + j * 0.24
    bottom = 0.08
    width = 0.2
    height = 0.18

    # Create the subplot
    ax_small = plt.axes([left, bottom, width, height])

    # Plot the KDE
    ax_small.plot(x_range, kde(x_range), 'b-')

    # Mark the excluded point
    ax_small.scatter([test_point], [0], color='red', s=80, marker='o',
                     label='Left-out point')

    # Mark all points
    ax_small.scatter(data, np.zeros_like(data), c='black', marker='|', s=50)

    # Set consistent y limits
    ax_small.set_ylim(0, y_max)

    # Set title
    ax_small.set_title(f'Left out x={test_point:.1f}')

    # Remove most labels for clarity
    if j > 0:
        ax_small.set_yticklabels([])

    ax_small.grid(True, alpha=0.3)

plt.figtext(0.5, 0.28, "Examples of Leave-One-Out KDEs with Optimal Bandwidth Factor",
            ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()'''

import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
from scipy.stats import gaussian_kde  # noqa

# Sample data
data = np.array([2.1, 2.3, 2.2, 2.7, 3.0, 3.1, 2.5, 2.8, 2.9, 3.2, 3.5, 3.7, 3.6, 4.0, 4.2, 4.1])

# Define a range of bandwidths to test (using scipy's bw_factor, which is a multiplier to Scott's rule)
bw_factors = np.logspace(-1, 1, 20)  # From 0.1 to 10 times Scott's rule


# Define our LOOCV function for KDE bandwidth
def kde_loocv(data, bw_factors):
    n = len(data)
    cv_scores = []

    # For each bandwidth factor
    for bw_factor in bw_factors:
        fold_scores = []

        # For each point in the dataset (leave-one-out)
        for i in range(n):
            # Training set: all points except i
            train_data = np.delete(data, i)

            # Test point
            test_point = np.array([data[i]])

            # Fit KDE on training data with current bandwidth factor
            kde = gaussian_kde(train_data, bw_method=bw_factor)

            # Evaluate log-density at the test point
            log_density = np.log(kde(test_point)[0])
            fold_scores.append(log_density)

        # Average log-density for this bandwidth
        cv_scores.append(np.mean(fold_scores))

    return cv_scores