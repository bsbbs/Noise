import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from os.path import join as join
svdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject/pyResults'
svdir = r'C:\Users\Bo\Dropbox (NYU Langone Health)\Bo Shen Working files\NoiseProject\pyResults'
# Global parameters preset

# Distribution demo
# parameter preset
color1 = 'orange'
color2 = 'red'
color3 = 'green'

# Divisive normalization
# Early noise
V1mean = 82
V2mean = 88
V3demo = np.array([0, .6, .9, 1])*(V1mean - 3)
Test = 'Early'
for version in ['additive', 'mean-scaled']:
    if version == 'additive':
        eps = 2.8
        eta = 0
    elif version == 'mean-scaled':
        eps = .1
        eta = 1e-2
    V1 = [V1mean, eps]
    V2 = [V2mean, eps]
    fig, axs = plt.subplots(4, 1, figsize=(3, 4))
    for frame in range(len(V3demo)):
        V3 = [V3demo[frame], eps]
        SVs, _ = DN(V1, V2, V3, eta, version)
        pdfs, x = getpdfs(SVs)
        adddistrilines(axs[frame], pdfs, x)
    plt.savefig(join(svdir, 'ModelSimulation', f'Demo_DN_{Test}_{version}.pdf'), format='pdf')
# Late noise
Test = 'Late'
eps = 0
eta = .9
V1 = [V1mean, eps]
V2 = [V2mean, eps]
fig, axs = plt.subplots(4, 1, figsize=(3, 4))
for frame in range(len(V3demo)):
    V3 = [V3demo[frame], eps]
    SVs, _ = DN(V1, V2, V3, eta, 'additive')
    pdfs, x = getpdfs(SVs)
    adddistrilines(axs[frame], pdfs, x)
plt.savefig(join(svdir, 'ModelSimulation', f'Demo_DN_{Test}.pdf'), format='pdf')
# Absolute model
V1mean = 82*.43
V2mean = 88*.43
V3demo = np.array([0, .6, .9, 1])*(V1mean - 3*.43)
# Early noise
Test = 'Early'
for version in ['additive', 'mean-scaled']:
    if version == 'additive':
        eps = 1.55
        eta = 0
    elif version == 'mean-scaled':
        eps = .07
        eta = 1e-2
    V1 = [V1mean, eps]
    V2 = [V2mean, eps]
    fig, axs = plt.subplots(4, 1, figsize=(3, 4))
    for frame in range(len(V3demo)):
        V3 = [V3demo[frame], eps]
        SVs, _ = Absolute(V1, V2, V3, eta, version)
        pdfs, x = getpdfs(SVs)
        adddistrilines(axs[frame], pdfs, x)
    plt.savefig(join(svdir, 'ModelSimulation', f'Demo_Absolute_{version}_{Test}.pdf'), format='pdf')
# Late noise
Test = 'Late'
eps = 0
eta = .9
V1 = [V1mean, eps]
V2 = [V2mean, eps]
fig, axs = plt.subplots(4, 1, figsize=(3, 4))
for frame in range(len(V3demo)):
    V3 = [V3demo[frame], eps]
    SVs, _ = Absolute(V1, V2, V3, eta, 'additive')
    pdfs, x = getpdfs(SVs)
    adddistrilines(axs[frame], pdfs, x)
plt.savefig(join(svdir, 'ModelSimulation', f'Demo_Absolute_{Test}.pdf'), format='pdf')

# Linear Subtraction model
V1mean = 84*.7
V2mean = 88*.7
V3demo = np.array([0, .6, .9, 1])*(V1mean - 2*.7)
# Early noise
Test = 'Early'
for version in ['additive', 'mean-scaled']:
    if version == 'additive':
        eps = 1.55
        eta = 0
    elif version == 'mean-scaled':
        eps = .04
        eta = 1e-2
    V1 = [V1mean, eps]
    V2 = [V2mean, eps]
    fig, axs = plt.subplots(4, 1, figsize=(3, 4))
    for frame in range(len(V3demo)):
        V3 = [V3demo[frame], eps]
        SVs, _ = LS(V1, V2, V3, eta, version)
        pdfs, x = getpdfs(SVs)
        adddistrilines(axs[frame], pdfs, x)
    plt.savefig(join(svdir, 'ModelSimulation', f'Demo_LinearSubtract_{version}_{Test}.pdf'), format='pdf')
# Late noise
Test = 'Late'
eps = 0
eta = .9
V1 = [V1mean, eps]
V2 = [V2mean, eps]
fig, axs = plt.subplots(4, 1, figsize=(3, 4))
for frame in range(len(V3demo)):
    V3 = [V3demo[frame], eps]
    SVs, _ = LS(V1, V2, V3, eta, 'additive')
    pdfs, x = getpdfs(SVs)
    adddistrilines(axs[frame], pdfs, x)
plt.savefig(join(svdir, 'ModelSimulation', f'Demo_LinearSubtract_{Test}.pdf'), format='pdf')

# Ratio curves
# Divisive normalization
V1mean = 150
V2mean = 158
Test = 'Early'
version = 'additive'
eps = 13
eta = 0
PlotAUCRatio(DN, Test, version, V1mean, V2mean, eps, eta, svdir)
version = 'mean-scaled'
eps = 1.25
eta = 0
PlotAUCRatio(DN, Test, version, V1mean, V2mean, eps, eta, svdir)
Test = 'Late'
version = 'additive'
eps = 0
eta = 3
PlotAUCRatio(DN, Test, version, V1mean, V2mean, eps, eta, svdir)
def PlotAUCRatio(func, Test, version, V1mean, V2mean, eps, eta, svdir):
    V3curves = np.arange(0, V2mean - 1, 1)
    V1 = [V1mean, eps]  # mean and std
    V2 = [V2mean, eps]
    ratio = []
    AUCval = []
    for V3mean in V3curves:
        V3 = [V3mean, eps]
        SVs, probs = func(V1, V2, V3, eta, version)
        ratio.append(probs[1] / (probs[0] + probs[1]) * 100)
        pdfs, x = getpdfs(SVs[:2])
        tmp = AUC(pdfs, x)
        AUCval.append(tmp)
    fig, axs = plt.subplots(2, 1, figsize=(3, 4))
    cmap = plt.get_cmap('viridis')
    # AUC
    axs[0].set_title(f'{Test} noise')
    axs[0].scatter(V3curves, AUCval, c=AUCval, cmap=cmap, marker='.', s=30)
    axs[0].set_xlim((-4, V3curves.max() + 6))
    axs[0].set_xlabel('')
    axs[0].set_ylabel('% Overlapping (V1 & V2)')
    axs[0].tick_params(axis='both', direction='in')
    # Ratio
    axs[1].scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
    axs[1].plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
    axs[1].plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
    axs[1].set_xlim((-4, V3curves.max() + 6))
    axs[1].set_xlabel('V3')
    axs[1].set_ylabel('% Correct (V1 & V2)')
    axs[1].tick_params(axis='both', direction='in')
    # plt.tight_layout()
    plt.savefig(join(svdir, 'ModelSimulation', f'AUCRatios_{func.__name__}_{Test}_{version}.pdf'), format='pdf')

# Divisive normalization
V1mean = 150
V2mean = 158
V3curves = np.arange(0, V2mean - 1, 1)
Test = 'Early'
for version in ['additive', 'mean-scaled']:
    if version == 'additive':
        eps = 13
    elif version == 'mean-scaled':
        eps = 1.25
    eta = 0
    V1 = [V1mean, eps]  # mean and std
    V2 = [V2mean, eps]
    ratio = []
    AUCval = []
    for V3mean in V3curves:
        V3 = [V3mean, eps]
        SVs, probs = DN(V1, V2, V3, eta, version)
        ratio.append(probs[1] / (probs[0] + probs[1])*100)
        pdfs, x = getpdfs(SVs[:2])
        tmp = AUC(pdfs, x)
        AUCval.append(tmp)
    #fig = plt.figure(figsize=(3, 2))
    fig, axs = plt.subplots(2, 1, figsize=(3, 4))
    cmap = plt.get_cmap('viridis')
    # AUC
    axs[0].set_title(f'{Test} noise')
    axs[0].scatter(V3curves, AUCval, c=AUCval, cmap=cmap, marker='.', s=30)
    axs[0].set_xlim((-4, V3curves.max() + 6))
    axs[0].set_xlabel('')
    axs[0].set_ylabel('% Overlapping (V1 & V2)')
    axs[0].tick_params(axis='both', direction='in')
    # Ratio
    axs[1].scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
    axs[1].plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
    axs[1].plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
    axs[1].set_xlim((-4, V3curves.max() + 6))
    axs[1].set_xlabel('V3')
    axs[1].set_ylabel('% Correct (V1 & V2)')
    axs[1].tick_params(axis='both', direction='in')
    #plt.tight_layout()
    plt.savefig(join(svdir, 'ModelSimulation', f'AUCRatios_DN_{Test}_{version}.pdf'), format='pdf')
Test = 'Late'
fig = plt.figure(figsize=(3, 2))
eps = 0
eta = 3
V1 = [V1mean, eps] # mean and std
V2 = [V2mean, eps]
ratio = []
for V3mean in V3curves:
    V3 = [V3mean, eps]
    SVs, probs = DN(V1, V2, V3, eta, 'additive')
    ratio.append(probs[1] / (probs[0] + probs[1])*100)
cmap = plt.get_cmap('viridis')
plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
plt.xlim((-4, V3curves.max() + 6))
plt.xlabel('V3')
plt.ylabel('% Correct (V1 & V2)')
plt.title(f'{Test} noise')
plt.tick_params(axis='both', direction='in')
plt.tight_layout()
plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_DN_{Test}.pdf'), format='pdf')
# Mixed noise
Test = 'Mixed'
V1mean = 150 # 45
V2mean = 158 # 58
# version = 'additive' #'mean-scaled' # 'additive'
for version in ['additive', 'mean-scaled']:
    if version == 'additive':
        # epsvec = np.linspace(0, 16.75, 8)
        epsvec = np.linspace(0, 13, 8)
    elif version == 'mean-scaled':
        # epsvec = np.linspace(0, 6.2, 8)
        epsvec = np.linspace(0, 1.25, 8)
    # etavec = np.linspace(12, 0, 8)
    etavec = np.linspace(3, 0, 8)
    gcolors = sns.color_palette('Spectral', 8)
    fig = plt.figure(figsize=(4.6, 3))
    for frame in range(len(epsvec)):
        eps = epsvec[frame]
        eta = etavec[frame]
        V1 = [V1mean, eps] # mean and std
        V2 = [V2mean, eps]
        #V3curves = np.linspace(0, V2mean-1, 11)
        V3curves = np.arange(0, V2mean - 1, 1)
        ratio = []
        for V3mean in V3curves:
            V3 = [V3mean, eps]
            SVs, probs = DN(V1, V2, V3, eta, version)
            ratio.append(probs[1] / (probs[0] + probs[1])*100)
        plt.plot(V3curves, ratio, '.-', color=gcolors[frame], linewidth=1, label=f"E{eps:.1f}, L{eta:.1f}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
    plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
    plt.xlim((-4, V3curves.max() + 6))
    plt.xlabel('V3')
    plt.ylabel('% Correct (V1 & V2)')
    plt.title('Relative accuracy')
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_DN_{Test}_{version}.pdf'), format='pdf')

# Absolute values
V1mean = 150
V2mean = 158
V3curves = np.arange(0, V2mean - 1, 1)
Test = 'Early'
for version in ['additive', 'mean-scaled']:
    if version == 'additive':
        eps = 13
    elif version == 'mean-scaled':
        eps = 1.1
    fig = plt.figure(figsize=(3, 2))
    eta = 0
    V1 = [V1mean, eps]  # mean and std
    V2 = [V2mean, eps]
    ratio = []
    for V3mean in V3curves:
        V3 = [V3mean, eps]
        SVs, probs = Absolute(V1, V2, V3, eta, version)
        ratio.append(probs[1] / (probs[0] + probs[1])*100)
    cmap = plt.get_cmap('viridis')
    plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
    plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
    plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
    plt.xlim((-4, V3curves.max() + 6))
    plt.xlabel('V3')
    plt.ylabel('% Correct (V1 & V2)')
    plt.title(f'{Test} noise')
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_Absolute_{Test}_{version}.pdf'), format='pdf')
Test = 'Late'
fig = plt.figure(figsize=(3, 2))
eps = 0
eta = 13
V1 = [V1mean, eps] # mean and std
V2 = [V2mean, eps]
ratio = []
for V3mean in V3curves:
    V3 = [V3mean, eps]
    SVs, probs = Absolute(V1, V2, V3, eta, 'additive')
    ratio.append(probs[1] / (probs[0] + probs[1])*100)
cmap = plt.get_cmap('viridis')
plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
plt.xlim((-4, V3curves.max() + 6))
plt.xlabel('V3')
plt.ylabel('% Correct (V1 & V2)')
plt.title(f'{Test} noise')
plt.tick_params(axis='both', direction='in')
plt.tight_layout()
plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_Absolute_{Test}.pdf'), format='pdf')

# Linear subtraction
V1mean = 150
V2mean = 158
V3curves = np.arange(0, V2mean - 1, 1)
Test = 'Early'
for version in ['additive', 'mean-scaled']:
    if version == 'additive':
        eps = 13
    elif version == 'mean-scaled':
        eps = 1.1
    fig = plt.figure(figsize=(3, 2))
    eta = 0
    V1 = [V1mean, eps]  # mean and std
    V2 = [V2mean, eps]
    ratio = []
    for V3mean in V3curves:
        V3 = [V3mean, eps]
        SVs, probs = LS(V1, V2, V3, eta, version)
        ratio.append(probs[1] / (probs[0] + probs[1])*100)
    cmap = plt.get_cmap('viridis')
    plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
    plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
    plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
    plt.xlim((-4, V3curves.max() + 6))
    plt.xlabel('V3')
    plt.ylabel('% Correct (V1 & V2)')
    plt.title(f'{Test} noise')
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_LinearSubtract_{Test}_{version}.pdf'), format='pdf')
Test = 'Late'
fig = plt.figure(figsize=(3, 2))
eps = 0
eta = 13
V1 = [V1mean, eps] # mean and std
V2 = [V2mean, eps]
ratio = []
for V3mean in V3curves:
    V3 = [V3mean, eps]
    SVs, probs = LS(V1, V2, V3, eta, 'additive')
    ratio.append(probs[1] / (probs[0] + probs[1])*100)
cmap = plt.get_cmap('viridis')
plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
plt.xlim((-4, V3curves.max() + 6))
plt.xlabel('V3')
plt.ylabel('% Correct (V1 & V2)')
plt.title(f'{Test} noise')
plt.tick_params(axis='both', direction='in')
plt.tight_layout()
plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_LinearSubtract_{Test}.pdf'), format='pdf')

# Overlapping curves
# Divisive normalization
V1mean = 150
V2mean = 158
# V3curves = np.linspace(0, V2mean-1, 71)
V3curves = np.arange(0, V2mean - 1, 1)
Test = 'Early'
for version in ['additive', 'mean-scaled']:
    if version == 'additive':
        eps = 13
    elif version == 'mean-scaled':
        eps = 1.25
    fig = plt.figure(figsize=(3, 2))
    eta = 0
    V1 = [V1mean, eps]  # mean and std
    V2 = [V2mean, eps]
    ratio = []
    for V3mean in V3curves:
        V3 = [V3mean, eps]
        SVs, probs = DN(V1, V2, V3, eta, version)
        V3 = [V3curves[frame], eps]
        SVs, probs = DN(V1, V2, V3, eta)
        pdfs, x = getpdfs(SVs)
        AUCval =  AUC(pdfs, x)

        # AUC
        AUCPlot(ax[1], SVs, V3curves, AUCrng, colors, frame)

        ratio.append(probs[1] / (probs[0] + probs[1])*100)
    cmap = plt.get_cmap('viridis')
    plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
    plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
    plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
    plt.xlim((-4, V3curves.max() + 6))
    plt.xlabel('V3')
    plt.ylabel('% Overlapping (V1 & V2)')
    plt.title(f'{Test} noise')
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.savefig(join(svdir, 'ModelSimulation', f'Overlap_DN_{Test}_{version}.pdf'), format='pdf')



Test = 'Late'
fig = plt.figure(figsize=(3, 2))
eps = 0
eta = 3
V1 = [V1mean, eps] # mean and std
V2 = [V2mean, eps]
ratio = []
for V3mean in V3curves:
    V3 = [V3mean, eps]
    SVs, probs = DN(V1, V2, V3, eta, 'additive')
    ratio.append(probs[1] / (probs[0] + probs[1])*100)
cmap = plt.get_cmap('viridis')
plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
plt.xlim((-4, V3curves.max() + 6))
plt.xlabel('V3')
plt.ylabel('% Correct (V1 & V2)')
plt.title(f'{Test} noise')
plt.tick_params(axis='both', direction='in')
plt.tight_layout()
plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_DN_{Test}.pdf'), format='pdf')

# Absolute values
V1mean = 150
V2mean = 158
V3curves = np.arange(0, V2mean - 1, 1)
Test = 'Early'
for version in ['additive', 'mean-scaled']:
    if version == 'additive':
        eps = 13
    elif version == 'mean-scaled':
        eps = 1.1
    fig = plt.figure(figsize=(3, 2))
    eta = 0
    V1 = [V1mean, eps]  # mean and std
    V2 = [V2mean, eps]
    ratio = []
    for V3mean in V3curves:
        V3 = [V3mean, eps]
        SVs, probs = Absolute(V1, V2, V3, eta, version)
        ratio.append(probs[1] / (probs[0] + probs[1])*100)
    cmap = plt.get_cmap('viridis')
    plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
    plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
    plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
    plt.xlim((-4, V3curves.max() + 6))
    plt.xlabel('V3')
    plt.ylabel('% Correct (V1 & V2)')
    plt.title(f'{Test} noise')
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_Absolute_{Test}_{version}.pdf'), format='pdf')
Test = 'Late'
fig = plt.figure(figsize=(3, 2))
eps = 0
eta = 13
V1 = [V1mean, eps] # mean and std
V2 = [V2mean, eps]
ratio = []
for V3mean in V3curves:
    V3 = [V3mean, eps]
    SVs, probs = Absolute(V1, V2, V3, eta, 'additive')
    ratio.append(probs[1] / (probs[0] + probs[1])*100)
cmap = plt.get_cmap('viridis')
plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
plt.xlim((-4, V3curves.max() + 6))
plt.xlabel('V3')
plt.ylabel('% Correct (V1 & V2)')
plt.title(f'{Test} noise')
plt.tick_params(axis='both', direction='in')
plt.tight_layout()
plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_Absolute_{Test}.pdf'), format='pdf')

# Linear subtraction
V1mean = 150
V2mean = 158
V3curves = np.arange(0, V2mean - 1, 1)
Test = 'Early'
for version in ['additive', 'mean-scaled']:
    if version == 'additive':
        eps = 13
    elif version == 'mean-scaled':
        eps = 1.1
    fig = plt.figure(figsize=(3, 2))
    eta = 0
    V1 = [V1mean, eps]  # mean and std
    V2 = [V2mean, eps]
    ratio = []
    for V3mean in V3curves:
        V3 = [V3mean, eps]
        SVs, probs = LS(V1, V2, V3, eta, version)
        ratio.append(probs[1] / (probs[0] + probs[1])*100)
    cmap = plt.get_cmap('viridis')
    plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
    plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
    plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
    plt.xlim((-4, V3curves.max() + 6))
    plt.xlabel('V3')
    plt.ylabel('% Correct (V1 & V2)')
    plt.title(f'{Test} noise')
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_LinearSubtract_{Test}_{version}.pdf'), format='pdf')
Test = 'Late'
fig = plt.figure(figsize=(3, 2))
eps = 0
eta = 13
V1 = [V1mean, eps] # mean and std
V2 = [V2mean, eps]
ratio = []
for V3mean in V3curves:
    V3 = [V3mean, eps]
    SVs, probs = LS(V1, V2, V3, eta, 'additive')
    ratio.append(probs[1] / (probs[0] + probs[1])*100)
cmap = plt.get_cmap('viridis')
plt.scatter(V3curves, ratio, c=ratio, cmap=cmap, marker='.', s=30)
plt.plot(V1mean, min(ratio), 'v', color=color1, markersize=4, alpha=1)
plt.plot(V2mean, min(ratio), 'v', color=color2, markersize=4, alpha=1)
plt.xlim((-4, V3curves.max() + 6))
plt.xlabel('V3')
plt.ylabel('% Correct (V1 & V2)')
plt.title(f'{Test} noise')
plt.tick_params(axis='both', direction='in')
plt.tight_layout()
plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_LinearSubtract_{Test}.pdf'), format='pdf')



# Alternative Model - Divisive normalization, with additive early noise (when version == 'additive') or with mean-scaled noise (version == 'mean-scaled')
def DN(V1, V2, V3, eta, version):
    w = 1
    M = 1
    Rmax = 75
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use
    options = torch.tensor([V1, V2, V3])
    num_samples = int(1e6)

    # Operation #1, sampling with early noise
    if version == 'additive':
        O1 = torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                          for mean, sd in options], dim=0)
        # Operation #2, G neurons independently summarizing these inputs with noise
        G1 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options], dim=0), dim=0)
        G2 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options], dim=0), dim=0)
        G3 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options], dim=0), dim=0)
    elif version == 'mean-scaled':
        O1 = torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                          for mean, slp in options], dim=0)
        # Operation #2, G neurons independently summarizing these inputs with noise
        G1 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options], dim=0), dim=0)
        G2 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options], dim=0), dim=0)
        G3 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options], dim=0), dim=0)

    # Operation #3, implementing lateral inhibition
    Context = torch.stack((G1, G2, G3), dim=0)
    O3 = [Rmax*DirectValue/(M+ContextValue*w) for DirectValue, ContextValue in zip(O1, Context)]
    # Alternatively, use the shared denominator by assuming noise varying trial-by-trial but fixed within trial
    # D = torch.sum(O1, dim=0)
    # O3 = [Rmax * DirectValue / (M + D * w) for DirectValue in O1]

    # Operation #4, apply late noise
    Outputs = torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device)
                           for ComputedValue in O3])
    max_indices = torch.argmax(Outputs, dim=0)
    max_from_each_distribution = torch.zeros_like(Outputs)
    max_from_each_distribution[max_indices, torch.arange(Outputs.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / Outputs.shape[1]
    return Outputs.cpu().numpy(), probs.cpu().numpy()
# Alternative Model - Absolute values, with additive early noise (when version == 'additive') or with mean-scaled noise (version == 'mean-scaled')
def Absolute(V1, V2, V3, eta, version):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use
    options = torch.tensor([V1, V2, V3])
    num_samples = int(1e6)

    # Operation #1, sampling with early noise
    if version == 'additive':
        O1 = torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                          for mean, sd in options], dim=0)
    elif version == 'mean-scaled':
        O1 = torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                          for mean, slp in options], dim=0)
    # Operation #4, apply late noise
    Outputs = torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device)
                           for ComputedValue in O1])
    max_indices = torch.argmax(Outputs, dim=0)
    max_from_each_distribution = torch.zeros_like(Outputs)
    max_from_each_distribution[max_indices, torch.arange(Outputs.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / Outputs.shape[1]
    return Outputs.cpu().numpy(), probs.cpu().numpy()
# Alternative Model - Linear subtraction, with additive early noise (when version == 'additive') or with mean-scaled noise (version == 'mean-scaled')
def LS(V1, V2, V3, eta, version):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use
    options = torch.tensor([V1, V2, V3])
    num_samples = int(1e6)
    w = .2

    # Operation #1, sampling with early noise
    if version == 'additive':
        O1 = torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                          for mean, sd in options], dim=0)
        # Operation #2, G neurons independently summarizing these inputs with noise
        G1 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options], dim=0), dim=0)
        G2 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options], dim=0), dim=0)
        G3 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options], dim=0), dim=0)
    elif version == 'mean-scaled':
        O1 = torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                          for mean, slp in options], dim=0)
        # Operation #2, G neurons independently summarizing these inputs with noise
        G1 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options], dim=0), dim=0)
        G2 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options], dim=0), dim=0)
        G3 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options], dim=0), dim=0)
    Context = torch.stack((G1, G2, G3), dim=0)
    # Operation #3, implementing lateral inhibition
    O3 = [DirectValue - ContextValue*w for DirectValue, ContextValue in zip(O1, Context)]
    # Operation #4, apply late noise
    Outputs = torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device)
                           for ComputedValue in O3])
    max_indices = torch.argmax(Outputs, dim=0)
    max_from_each_distribution = torch.zeros_like(Outputs)
    max_from_each_distribution[max_indices, torch.arange(Outputs.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / Outputs.shape[1]
    return Outputs.cpu().numpy(), probs.cpu().numpy()
def makecolors(values):
    norm = Normalize(vmin=values.min(), vmax=values.max())
    normalized_values = norm(values)
    cmap = plt.get_cmap('viridis')
    colors = cmap(normalized_values)
    return colors
def getpdfs(SVs):
    # ys = [np.arange(np.mean(SV) - 3 * np.std(SV), np.mean(SV) + 3 * np.std(SV), .1) for SV in SVs]
    # pdfs = [stats.norm.pdf(y, np.mean(SV), np.std(SV)) for y, SV in zip(ys, SVs)]
    # kdes = [stats.gaussian_kde(SV) for SV in SVs]
    # pdfs = [kde(y) for kde, y in zip(kdes, ys)]
    interval = 0.1
    xlim = 65
    x = np.arange(0, xlim, interval)
    kdes = [stats.gaussian_kde(SV) for SV in SVs]
    pdfs = [kde(x) for kde in kdes]
    return pdfs, x
def adddistrilines(ax, pdfs, x):
    mask = pdfs[0] > 1e-4
    ax.plot(x[mask], pdfs[0][mask], label='Option 1', color=color1)
    mask = pdfs[1] > 1e-4
    ax.plot(x[mask], pdfs[1][mask],  label='Option 2', color=color2)
    mask = pdfs[2] > 1e-4
    ax.plot(x[mask], pdfs[2][mask],  label='Option 3', color=color3, alpha=0.5, zorder=1)

    overlap = np.minimum(pdfs[0], pdfs[1])
    cutout = np.minimum(overlap, pdfs[2])
    mask = overlap > 1e-4
    ax.fill_between(x[mask], cutout[mask], overlap[mask], color='darkslategrey', label='Overlap Area')
    ax.fill_between(x[mask], 0, cutout[mask], color='darkslategrey', alpha=0.5, label='Overlap Area')

    ax.set_xlim(0, 45)
    ax.set_xticks([0, 20, 40])
    ax.set_ylim(0, .5)
    ax.set_yticks([0, .5])
    ax.set_xlabel('Neural activity')
    ax.set_ylabel('Probability density')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', direction='in')
def AUC(pdfs, x):
    interval = x[1] - x[0]
    overlap = np.minimum(pdfs[0], pdfs[1])
    AUCval = sum(overlap) * interval * 100
    return AUCval





fig = plt.figure(figsize=(3.2, 3))
colors = ['blue', 'red', 'lightblue', 'pink']
eta = 13
slp = 2.52
V1 = [V1mean, slp]  # mean and the slope of mean-scaled variance
V2 = [V2mean, slp]
V3curves = np.arange(0, V1mean, 1)
ratio = []
for V3mean in V3curves:
    V3 = [V3mean, slp]
    SVs, probs = DN(V1, V2, V3, eta, 'mean-scaled')
    ratio.append(probs[1] / (probs[0] + probs[1]) * 100)
plt.scatter(V3curves, ratio, color='red', marker='.', s=30)
plt.xlim((-3, V3curves.max() + 3))
plt.xlabel('V3')
plt.ylabel('% Correct (V1 & V2)')
plt.title('Relative accuracy')
plt.tight_layout()

# 2-by-2 design
V1mean = 45
V2mean = 58
rws = np.arange(0, 1.1, 0.1)
colors = makecolors(rws)
fig = plt.figure(figsize=(3.2, 3))
#cmap = plt.get_cmap('viridis')
for i, rw in enumerate(rws):
    amp = 16
    eta = rw*amp# 17#3#3#17/2#13#13#13#0#4#0 - late noise
    eps = (1-rw)*amp#13#13#17#17/2#0#17#0#17#0#5 - early noise
    V1 = [V1mean, eps]  # mean and std
    V2 = [V2mean, eps]
    V3curves = np.arange(0, V1mean, 3)
    ratio = []
    for V3mean in V3curves:
        V3 = [V3mean, eps]
        SVs, probs = DN(V1, V2, V3, eta)
        ratio.append(probs[1] / (probs[0] + probs[1]) * 100)
    #plt.scatter(V3curves, ratio, color=, marker='.-', s=30)
    plt.scatter(V3curves, ratio, color=colors[i], marker='.', s=30)
plt.xlim((-3, V3curves.max() + 3))
plt.xlabel('V3')
plt.ylabel('% Correct (V1 & V2)')
plt.title('Relative accuracy')
plt.tight_layout()

fig = plt.figure(figsize=(3.2, 3))
colors = ['blue', 'red', 'lightblue', 'pink']
Leta = 1#14
Seta = 1#5
Leps = 1#16
Seps = 1#7
etas = [Seta, Seta, Leta, Leta] # late noise
epss = [Leps, Seps, Leps, Seps] # early noise
for i in range(4):
    eta = etas[i]
    eps = epss[i]
    V1 = [V1mean, eps]  # mean and std
    V2 = [V2mean, eps]
    V3curves = np.arange(0, V1mean, 3)
    ratio = []
    for V3mean in V3curves:
        V3 = [V3mean, eps]
        SVs, probs = DN(V1, V2, V3, eta)
        ratio.append(probs[1] / (probs[0] + probs[1]) * 100)
    plt.scatter(V3curves, ratio, color=colors[i], marker='.', s=30)
plt.xlim((-3, V3curves.max() + 3))
plt.xlabel('V3')
plt.ylabel('% Correct (V1 & V2)')
plt.title('Relative accuracy')
plt.tight_layout()


# Late noise only
# Test = 'Late'
# eps=0
# eta=3 # 1.7
Test = "Early"
eps=5
eta=0
V1=[V1mean, eps] # mean and std
V2=[V2mean, eps]

def set_axis(ax):
    ax.set_yticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.annotate('', xy=(0, 0), xytext=(0, ylim),
                arrowprops=dict(arrowstyle='<-', linewidth=1.5, color='grey'))
    ax.annotate('', xy=(xlim, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', linewidth=1.5, color='grey'))

def getpdfs(SVs):
    # ys = [np.arange(np.mean(SV) - 3 * np.std(SV), np.mean(SV) + 3 * np.std(SV), .1) for SV in SVs]
    # pdfs = [stats.norm.pdf(y, np.mean(SV), np.std(SV)) for y, SV in zip(ys, SVs)]
    # kdes = [stats.gaussian_kde(SV) for SV in SVs]
    # pdfs = [kde(y) for kde, y in zip(kdes, ys)]
    interval = 0.1
    y = np.arange(0, ylim, interval)
    kdes = [stats.gaussian_kde(SV) for SV in SVs]
    pdfs = [kde(y) for kde in kdes]
    return pdfs, y
def adddistrilines(ax, SVs, pdfs, y, V1, V2, V3, maxheight, frame, overlap0):
    height = 200
    mask = pdfs[0] > 1e-4
    ax.plot(-pdfs[0][mask] * height, y[mask], label='Option 1', color=color1)
    ax.plot([0, V1[0]], [np.mean(SVs[0]), np.mean(SVs[0])], color=color1, linestyle='--', linewidth=1)
    ax.plot([V1[0], V1[0]], [0, np.mean(SVs[0])], color=color1, linestyle='--', linewidth=1)
    mask = pdfs[1] > 1e-4
    ax.plot(-pdfs[1][mask] * height, y[mask], label='Option 2', color=color2)
    ax.plot([0, V2[0]], [np.mean(SVs[1]), np.mean(SVs[1])], color=color2, linestyle='--', linewidth=1)
    ax.plot([V2[0], V2[0]], [0, np.mean(SVs[1])], color=color2, linestyle='--', linewidth=1)
    # ax.plot(-pdfs[2] * height, ys[2], label='Option 3', color=color3, alpha=0.5, zorder=1)

    overlap = np.minimum(pdfs[0], pdfs[1])
    #cutout = np.minimum(overlap, pdf[2])
    mask = overlap > 1e-4
    ax.fill_between(np.concatenate([np.zeros(1), -overlap[mask]*height, np.zeros(1)]), np.append(np.append(y[mask].min(), y[mask]), y[mask].max()), color='orangered', edgecolor='none', label='Overlap Area')
    if (frame > 0):
        mask0 = overlap0 > 1e-4
        #ax.fill_between(-overlap0[mask0] * height, x[mask0], x[mask0], color='lightgray', label='Overlap Area')
        ax.fill_between(np.concatenate([np.zeros(1), -overlap0[mask0] * height, np.zeros(1)]),
                        np.append(np.append(y[mask0].min(), y[mask0]), y[mask0].max()), color='lightgray',
                        edgecolor='none', label='Overlap Area')
    if V1[1]>0:
        xs = [np.arange(meanval - 3 * sdval, meanval + 3 * sdval, .1) for meanval, sdval in [V1, V2, V3]]
        pdfxs = [stats.norm.pdf(np.arange(meanval - 3 * sdval, meanval + 3 * sdval, .1), meanval, sdval) for
                 meanval, sdval in [V1, V2, V3]]
    if V1[1]>0:
        ax.plot(xs[0], pdfxs[0] / pdfxs[0].max() * 5, label='V1', color=color1)
    if V2[1] > 0:
        ax.plot(xs[1], pdfxs[1] / pdfxs[0].max() * 5, label='V2', color=color2)
    #if V3[1] > 0:
        #ax.plot(xs[2], pdfxs[2] / pdfxs[0].max() * 5, label='V3', color=color3, alpha=0.5, zorder=1)
    ax.annotate('V3', (V3[0], -5), color=color3, fontsize=11,
                   ha='center', va='bottom')
    ax.spines['bottom'].set_position('zero')
    ax.set_xticks([V1[0], V2[0]], labels=['V1', 'V2'], color='orangered')
    ax.set_xlim(-maxheight, xlim)
    ax.set_ylim(-5, ylim)
    ax.set_xlabel('Inputs')
    ax.set_ylabel('Neural activity')  # adjust padding
def AUCPlot(ax, SVs, V3curves, AUCrng, colors, frame):
    interval = 0.1
    x = np.arange(0, ylim, interval)
    kdes = [stats.gaussian_kde(SV) for SV in SVs[:2]]
    pdfs = [kde(x) for kde in kdes]
    overlap = np.minimum(pdfs[0], pdfs[1])
    AUCval = sum(overlap) * interval * 100
    ax.plot(V3curves[frame], AUCval, '.', label="Overlapping", color=colors[frame])
    ax.set_xlabel('V3')
    ax.set_ylabel('% (V1 & V2)')  # adjust padding
    ax.set_ylim([AUCrng.min()-1, AUCrng.max()+1])
    ax.set_xlim([V3curves.min()-1,V3curves.max()+1])
    ax.set_title('Overlapping')

# Distribution and overlapping
#V3curves = np.append(np.linspace(0,((V1mean-3)/2)**0.5,25)**2, np.linspace(((V1mean-3)/2)**2, (V1mean-3)**2, 25)**0.5)
V3curves = np.append(np.linspace(0,(xlim/2)**0.5,25)**2, np.linspace((xlim/2)**2, xlim**2, 25)**0.5)
V3 = [V3curves.min(), eps]
SVs, _ = DN(V1, V2, V3, eta)
pdfs, y = getpdfs(SVs)
overlap0 = np.minimum(pdfs[0], pdfs[1])
V3 = [V3curves.max(), eps]
SVs, _ = DN(V1, V2, V3, eta)
pdfs, y = getpdfs(SVs)
maxheight = pdfs[1].max()*220

_, ax = plt.subplots(1,3)
fig = plt.figure(figsize=(3.2*3, 3))
# ax[0] = fig.add_axes([.05, .1, .3, .9])
ax[0] = fig.add_axes([0.03, .1, .35, .9])
def distrplot(frame):
    V3 = [V3curves[frame], eps]
    SVs, _ = DN(V1, V2, V3, eta)
    pdfs, y = getpdfs(SVs)
    ax[0].clear()
    set_axis(ax[0])
    adddistrilines(ax[0], SVs, pdfs, y, V1, V2, V3, maxheight, frame, overlap0)
    if (frame==0) | (frame==len(V3curves)-1):
        plt.savefig(join(svdir, 'ModelSimulation', f'DistriPlot_DN_{Test}{frame}.pdf'), format='pdf')
ani = FuncAnimation(fig, distrplot, frames=len(V3curves), repeat=False)
ani.save(join(svdir, 'ModelSimulation', f'DistriPlot_DN_{Test}.gif'), writer='pillow', fps=20)

V3curves = np.append(np.linspace(0,((V1mean-3)/2)**0.5,25)**2, np.linspace(((V1mean-3)/2)**2, (V1mean-3)**2, 25)**0.5)
AUCrng = np.array([AUC([V3mean, eps]) for V3mean in [V3curves.min(), V3curves.max()]])
colors = makecolors(V3curves)
_, ax = plt.subplots(1,3)
fig = plt.figure(figsize=(3.2*3, 3))
ax[0] = fig.add_axes([.03, .1, .35, .9])
ax[1] = fig.add_axes([.465, .2, .2, .6])
def OverlapPlot(frame):
    V3 = [V3curves[frame], eps]
    SVs, probs = DN(V1, V2, V3, eta)
    # AUC
    AUCPlot(ax[1], SVs, V3curves, AUCrng, colors, frame)
    # Distribution
    pdfs, y = getpdfs(SVs)
    ax[0].clear()
    set_axis(ax[0])
    adddistrilines(ax[0], SVs, pdfs, y, V1, V2, V3, maxheight, frame, overlap0)
    if (frame==0) | (frame==len(V3curves)-1):
        plt.savefig(join(svdir, 'ModelSimulation', f'OverlapPlot_DN_{Test}{frame}.pdf'), format='pdf')
ani = FuncAnimation(fig, OverlapPlot, frames=len(V3curves), repeat=False, blit=False)
ani.save(join(svdir, 'ModelSimulation', f'OverlapPlot_DN_{Test}.gif'), writer='pillow', fps=20)
