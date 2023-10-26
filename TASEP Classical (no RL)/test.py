import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scipy.stats as stats

@njit
def truncated_normal(mean, std_dev, size):
    samples = []
    while len(samples) < size:
        new_samples = np.random.normal(mean, std_dev, size)
        new_samples = new_samples[(new_samples >= 0) & (new_samples <= 1)]
        samples.extend(new_samples)
    return np.array(samples[:size])

# Generate samples


x_values = np.linspace(0, 1, 100)
clip0, clip1 = 0, 1
loc, scale = 0.5, 0.5
a, b = (clip0 - loc) / scale, (clip1 - loc) / scale
y_values = stats.truncnorm.pdf(x_values, a, b, loc, scale)
plt.plot(x_values, y_values, 'r')
samples = truncated_normal(loc, scale, 1000000)


# Plot histogram
plt.hist(samples, bins=50, density=True)
plt.title('Histogram of Truncated Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.plot
plt.show()