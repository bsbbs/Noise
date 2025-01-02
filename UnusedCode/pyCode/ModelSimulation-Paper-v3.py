import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from myFunctions import adddistrilines, getpdfs, dDN
from os.path import join as join
svdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject/pyResults'
# svdir = r'C:\Users\Bo\Dropbox (NYU Langone Health)\Bo Shen Working files\NoiseProject\pyResults'
# Global parameters preset

# Distribution Animation Demos
# parameter preset
# Divisive normalization
# Early noise
V1mean = 88
V2mean = 83
V3demo = np.array([0, .6, .9, 1])*(V2mean - 3)
Test = 'Early'
eps = 2.8
eta1 = 0
eta2 = 0
eta3 = 0
V1 = [V1mean, eps]
V2 = [V2mean, eps]
fig, axs = plt.subplots(4, 1, figsize=(3, 4))
for frame in range(len(V3demo)):
    V3 = [V3demo[frame], eps]
    SVs, _ = dDN(V1, V2, V3, eta1, eta2, eta3, version)
    pdfs, x = getpdfs(SVs)
    adddistrilines(axs[frame], pdfs, x)
plt.savefig(join(svdir, 'ModelSimulation', f'Demo_dDN_{Test}.pdf'), format='pdf')
# Late noise
Test = 'Late'
eps = 0
eta1 = 1.25
eta2 = 1.25
eta3 = 1.25
V1 = [V1mean, eps]
V2 = [V2mean, eps]
fig, axs = plt.subplots(4, 1, figsize=(3, 4))
for frame in range(len(V3demo)):
    V3 = [V3demo[frame], eps]
    SVs, _ = dDN(V1, V2, V3, eta1, eta2, eta3, 'additive')
    pdfs, x = getpdfs(SVs)
    adddistrilines(axs[frame], pdfs, x)
plt.savefig(join(svdir, 'ModelSimulation', f'Demo_dDN_{Test}.pdf'), format='pdf')

## Changing contextual variance
# Early noise
V1mean = 88
V2mean = 83
V3demo = .6*(V2mean - 3)
V3eps = np.array([2.2, 5.5, 13.2])
Test = 'Early_ContextVariance'
version = 'additive'
eps = 2.8
eta1 = 0
eta2 = 0
eta3 = 0
V1 = [V1mean, eps]
V2 = [V2mean, eps]
fig, axs = plt.subplots(3, 1, figsize=(2, 2.25))
for frame in range(len(V3eps)):
    V3 = [V3demo, V3eps[frame]]
    SVs, _ = dDN(V1, V2, V3, eta1, eta2, eta3, version)
    pdfs, x = getpdfs(SVs)
    adddistrilines(axs[frame], pdfs, x)
plt.savefig(join(svdir, 'ModelSimulation', f'Demo_dDN_{Test}.pdf'), format='pdf')

# Late noise
Test = 'Late_ContextVariance'
eps = 0
eta1 = 1.25
eta2 = 1.25
eta3s = np.array([.8, 2, 4.8])
V1 = [V1mean, eps]
V2 = [V2mean, eps]
V3demo = .6*(V2mean - 3)
V3 = [V3demo, eps]
fig, axs = plt.subplots(3, 1, figsize=(2, 2.25))
for frame in range(len(eta3s)):
    SVs, _ = dDN(V1, V2, V3, eta1, eta2, eta3s[frame], 'additive')
    pdfs, x = getpdfs(SVs)
    adddistrilines(axs[frame], pdfs, x)
plt.savefig(join(svdir, 'ModelSimulation', f'Demo_dDN_{Test}.pdf'), format='pdf')
