import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from myFunctions import load_data, PlotAUCRatio, adddistrilines, getpdfs, dDN, dDNb, dDNDpdnt, dDNwOpt, Absolute, LS
from matplotlib.colors import Normalize
from os.path import join as join
svdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject/pyResults'
# svdir = r'C:\Users\Bo\Dropbox (NYU Langone Health)\Bo Shen Working files\NoiseProject\pyResults'
# Global parameters preset

# Distribution demo
# parameter preset
# Divisive normalization
# Early noise
V1mean = 88
V2mean = 83
V3demo = np.array([0, .6, .9, 1])*(V2mean - 3)
Test = 'Early'
for version in ['additive']:#, 'mean-scaled'
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
        SVs, _ = dDN(V1, V2, V3, eta, version)
        pdfs, x = getpdfs(SVs)
        adddistrilines(axs[frame], pdfs, x)
    plt.savefig(join(svdir, 'ModelSimulation', f'Demo_dDN_{Test}_{version}.pdf'), format='pdf')
# Late noise
Test = 'Late'
eps = 0
eta = 1 #.9
V1 = [V1mean, eps]
V2 = [V2mean, eps]
fig, axs = plt.subplots(4, 1, figsize=(3, 4))
for frame in range(len(V3demo)):
    V3 = [V3demo[frame], eps]
    SVs, _ = dDN(V1, V2, V3, eta, 'additive')
    pdfs, x = getpdfs(SVs)
    adddistrilines(axs[frame], pdfs, x)
plt.savefig(join(svdir, 'ModelSimulation', f'Demo_dDN_{Test}.pdf'), format='pdf')

# Late noise
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
# Divisive normalization, dependent draw
V1mean = 150
V2mean = 158
Test = 'Early'
version = 'additive'
eps = 13
eta = 0
PlotAUCRatio(DN1, Test, version, V1mean, V2mean, eps, eta, svdir)
version = 'mean-scaled'
eps = 1.25
eta = 0
PlotAUCRatio(DN1, Test, version, V1mean, V2mean, eps, eta, svdir)
# Divisive normalization, semi-dependent draw
version = 'additive'
eps = 13
eta = 0
PlotAUCRatio(DN2, Test, version, V1mean, V2mean, eps, eta, svdir)
version = 'mean-scaled'
eps = 1.25
eta = 0
Test = 'Early_Asymtrc' # eps for V1 V2 = 1.25, eps for V3 = 0
eps = 0
PlotAUCRatio(DN2, Test, version, V1mean, V2mean, eps, eta, svdir)
# Divisive normalization, independent draw
version = 'additive'
eps = 13
eta = 0
PlotAUCRatio(DN3, Test, version, V1mean, V2mean, eps, eta, svdir)
version = 'mean-scaled'
eps = 1.25
eta = 0
Test = 'Early_Asymtrc' # eps V1 V2 = 1.25, eps V3 = 5
eps = 5 #1.25
PlotAUCRatio(DN3, Test, version, V1mean, V2mean, eps, eta, svdir)
Test = 'Late'
version = 'additive'
eps = 0
eta = 3
PlotAUCRatio(DN, Test, version, V1mean, V2mean, eps, eta, svdir)

# Absolute
V1mean = 150
V2mean = 158
Test = 'Early'
version = 'additive'
eps = 13
eta = 0
PlotAUCRatio(Absolute, Test, version, V1mean, V2mean, eps, eta, svdir)
version = 'mean-scaled'
eps = 1.1
eta = 0
PlotAUCRatio(Absolute, Test, version, V1mean, V2mean, eps, eta, svdir)
Test = 'Late'
version = 'additive'
eps = 0
eta = 13
PlotAUCRatio(Absolute, Test, version, V1mean, V2mean, eps, eta, svdir)

# Linear subtraction
V1mean = 150
V2mean = 158
Test = 'Early'
version = 'additive'
eps = 13
eta = 0
PlotAUCRatio(LS, Test, version, V1mean, V2mean, eps, eta, svdir)
version = 'mean-scaled'
eps = 1.1
eta = 0
PlotAUCRatio(LS, Test, version, V1mean, V2mean, eps, eta, svdir)
Test = 'Late'
version = 'additive'
eps = 0
eta = 13
PlotAUCRatio(LS, Test, version, V1mean, V2mean, eps, eta, svdir)


# Mixed noise - Applied for DN only
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
            SVs, probs = DNb(V1, V2, V3, eta, version)
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
    plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_DNcutoff_{Test}_{version}.pdf'), format='pdf')

# 2-by-2 design - Applied for DN only
Test = '2by2'
V1mean = 150 # 45
V2mean = 158 # 58
version = "mean-scaled" #'additive'
# early noise
if version == "mean-scaled":
    sE = .7
    lE = 2.5
elif version == "additive":
    sE = 4
    lE = 13
# late noise
sL = .8
lL = 3
epsvec = [lE, sE, lE, sE] # early noise
etavec = [sL, sL, lL, lL] # late noise
gcolors = ['blue', 'pink', 'lightblue', 'red']
fig = plt.figure(figsize=(4.6, 3))
for frame in range(len(epsvec)):
    eps = epsvec[frame]
    eta = etavec[frame]
    V1 = [V1mean, sE]
    V2 = [V2mean, sE]
    V3curves = np.linspace(0, V1mean*.8, 21)
    #V3curves = np.arange(0, V2mean - 1, 1)
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
plt.title(f'{Test}_{version}')
plt.tick_params(axis='both', direction='in')
plt.tight_layout()
plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_DN_{Test}_{version}.pdf'), format='pdf')


# 2by2 design, Match the data variance structure - Applied for DN only
Test = 'DataStruct'
AccVarV3 = load_data(r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/myData', 'AccVarV3scld.txt')
V1mean = 150 # 45
V2mean = 158 # 58
version = 'additive'
sL = 1
lL = 2
etavec = [sL, lL]
gcolors = ['blue', 'pink', 'lightblue', 'red']
fig = plt.figure(figsize=(4.6, 3))
frame = 0
for TimePressure in ['Low', 'High']:
    if TimePressure == 'High':
        eta = lL
    elif TimePressure == 'Low':
        eta = sL
    for Vagueness in ['Vague', 'Precise']:
        V1 = [V1mean, 13]
        V2 = [V2mean, 13]
        maskT = AccVarV3['TimePressure'] == TimePressure
        maskV = AccVarV3['Vagueness'] == Vagueness
        maskV3 = AccVarV3['V3scld'] < 1
        condat = AccVarV3[maskT & maskV & maskV3]
        V3scld = condat['V3scld']
        varbid3scld = condat['varbid3scld']
        ratio = []
        for v3i in condat.index:
            V3 = [V3scld[v3i]*V1mean, (varbid3scld[v3i]**.5)*V1mean]
            SVs, probs = DN(V1, V2, V3, eta, version)
            ratio.append(probs[1] / (probs[0] + probs[1])*100)
        plt.plot(V3scld, ratio, '.-', color=gcolors[frame], linewidth=1, label=f"E{Vagueness}, L{TimePressure}")
        frame += 1
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Scaled V3')
plt.ylabel('% Correct (V1 & V2)')
plt.tick_params(axis='both', direction='in')
plt.tight_layout()
plt.savefig(join(svdir, 'ModelSimulation', f'Ratios_DN_{Test}_{version}.pdf'), format='pdf')


