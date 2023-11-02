import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as join
svdir = r'C:\Users\Bo\Dropbox (NYU Langone Health)\CESS-Bo\pyResults'

# Parameters for the Gaussian distributions
mean = 0  # Mean of the distribution
std1 = 1   # Standard deviation of the distribution
std2 = 4
std3 = ((std1**2 + std2**2)/2)**.5
std4 = std3
size = 10000  # Number of elements in the array
# Initialize an empty list to store the choice
p_unequal = [] # probability of choosing Option 1
p_equal = []
mean_diff = np.arange(-100, 101, 1).astype(int)/10
Exam = {-10: 'Left', 0: 'Overlap', 10: 'Right'}
# Generate the distributions and calculate the differences
for md in mean_diff:
    gaussian_array1 = np.random.normal(mean+md, std1, size)
    gaussian_array2 = np.random.normal(mean, std2, size)
    gaussian_array3 = np.random.normal(mean+md, std3, size)
    gaussian_array4 = np.random.normal(mean, std4, size)
    d = (gaussian_array1 > gaussian_array2).astype(int)
    p_unequal.append(np.mean(d))
    d = (gaussian_array3 > gaussian_array4).astype(int)
    p_equal.append(np.mean(d))
    if md in Exam:
        # plot the distributions of the two
        bin_width = 0.5
        bins = np.arange(-25, 25, bin_width)
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 5))
        sns.set_palette('Paired')
        sns.histplot(gaussian_array2, bins=bins, kde=True, alpha=0.5, label='Option 2', ax=axs[0])
        sns.histplot(gaussian_array1, bins=bins, kde=True, alpha=0.5, label='Option 1', ax=axs[0])
        handles, labels = axs[0].get_legend_handles_labels()
        sns.histplot(gaussian_array4, bins=bins, kde=True, alpha=0.5, label='Option 2', ax=axs[1])
        sns.histplot(gaussian_array3, bins=bins, kde=True, alpha=0.5, label='Option 1', ax=axs[1])
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Unequal distributions')
        axs[0].legend(handles[::-1], labels[::-1])
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Equal distributions')
        plt.tight_layout()
        plt.show()
        plt.savefig(join(svdir, 'LogisticSimulation', 'Histograms_'+Exam[md]+'.pdf'), format='pdf')
# Plot the differences
# Bhattacharyya distance
var_sum = std1**2 + std2**2
coeff_unequal = 0.25 * mean_diff**2 / var_sum + 0.5 * np.log(var_sum / (2 * std1 * std2 + np.finfo(float).eps))
distance_unequal = np.sqrt(1 - np.exp(-coeff_unequal))
var_sum = std3**2 + std4**2
coeff_equal = 0.25 * mean_diff**2 / var_sum + 0.5 * np.log(var_sum / (2 * std3 * std4 + np.finfo(float).eps))
distance_equal = np.sqrt(1 - np.exp(-coeff_equal))
# Gaussian random sample probability
SZ_unequal = (std1**2 + std2**2)**.5
prob_unequal = 1 - norm.cdf(-mean_diff/SZ_unequal)
SZ_equal = (std3**2 + std4**2)**.5
prob_equal = 1 - norm.cdf(-mean_diff/SZ_equal)

sns.set_palette('pastel')
plt.figure(figsize=(7, 4))
sns.set_palette('Paired')
plt.plot(mean_diff, p_equal, '-', label='Simulated Choice - Equal noise')
plt.plot(mean_diff, p_unequal, '-', label='Simulated Choice - Unequal noise')
plt.plot(mean_diff, prob_equal, '--', label='Theoretical Prediction - Equal')
plt.plot(mean_diff, prob_unequal, '--', label='Theoretical Prediction - Unequal')
plt.plot(mean_diff, distance_equal, '-', label='Bhattacharyya distance - Equal')
plt.plot(mean_diff, distance_unequal, '-', label='Bhattacharyya distance - Unequal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Mean Value Difference')
plt.ylabel('P (choose 1)')
plt.title('Distribution with unequal noise')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(join(svdir, 'LogisticSimulation', 'PsychometricCurves.pdf'), format='pdf')
