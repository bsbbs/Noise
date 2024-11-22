import matplotlib
candidates = ['Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
for candy in candidates:
    try:
        matplotlib.use(candy)
        print('Using backend: ' + candy)
        break
    except (ImportError, ModuleNotFoundError):
        pass


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create example data
data = np.random.rand(10, 10)

# Plot heatmap using seaborn
sns.heatmap(data, cmap='coolwarm', annot=True, fmt=".2f")

# Show the plot
plt.show()
