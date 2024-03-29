import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from myFunctions import load_data
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
for version in ['additive']: # , 'mean-scaled'
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
    plt.savefig(join(svdir, 'ModelSimulation', f'Demo_DN2_{Test}_{version}.pdf'), format='pdf')
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
    if ((max(AUCval) - min(AUCval)) < 1):
        axs[0].set_ylim((np.mean(AUCval)-.9, np.mean(AUCval)+1.8))
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
def DN1(V1, V2, V3, eta, version):
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
    elif version == 'mean-scaled':
        O1 = torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                          for mean, slp in options], dim=0)
    # Operation #3, Values in the same trial are always the same
    Context = torch.sum(O1, dim=0)
    O3 = [Rmax*DirectValue/(M+Context*w) for DirectValue in O1]
    # Operation #4, apply late noise
    Outputs = torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device)
                           for ComputedValue in O3])
    max_indices = torch.argmax(Outputs, dim=0)
    max_from_each_distribution = torch.zeros_like(Outputs)
    max_from_each_distribution[max_indices, torch.arange(Outputs.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / Outputs.shape[1]
    return Outputs.cpu().numpy(), probs.cpu().numpy()
def DN2(V1, V2, V3, eta, version):
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
        # Operation #2, G neurons independently summarizing contextual inputs with noise
        G1 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options[[1, 2]]], dim=0), dim=0)
        G2 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options[[0, 2]]], dim=0), dim=0)
        G3 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options[[0, 1]]], dim=0), dim=0)
    elif version == 'mean-scaled':
        O1 = torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                          for mean, slp in options], dim=0)
        # Operation #2, G neurons independently summarizing these inputs with noise
        G1 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options[[1, 2]]], dim=0), dim=0)
        G2 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options[[0, 2]]], dim=0), dim=0)
        G3 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options[[0, 1]]], dim=0), dim=0)
    # Operation #3, values (numerator and denominator) of the same item is the same only in the same option
    Context = torch.stack((G1, G2, G3), dim=0)
    O3 = [Rmax*DirectValue/(M+ (DirectValue + ContextValue)*w) for DirectValue, ContextValue in zip(O1, Context)]
    # Operation #4, apply late noise
    Outputs = torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device)
                           for ComputedValue in O3])
    max_indices = torch.argmax(Outputs, dim=0)
    max_from_each_distribution = torch.zeros_like(Outputs)
    max_from_each_distribution[max_indices, torch.arange(Outputs.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / Outputs.shape[1]
    return Outputs.cpu().numpy(), probs.cpu().numpy()
def DN3(V1, V2, V3, eta, version):
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
    # Operation #3, every time access to the value of the item is different, even within the same trial and for the same option,
    # e.g, V1 in the numerator and in the denominator of option 1 are independently drawn
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
def DNb(V1, V2, V3, eta, version):
    w = 1
    M = 1
    Rmax = 75
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use
    options = torch.tensor([V1, V2, V3])
    num_samples = int(1e6)
    # Operation #1, sampling with early noise
    if version == 'additive':
        O1 = torch.clamp(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                          for mean, sd in options], dim=0), min=0)
        # Operation #2, G neurons independently summarizing these inputs with noise
        G1 = torch.sum(torch.clamp(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options], dim=0), min=0), dim=0)
        G2 = torch.sum(torch.clamp(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options], dim=0), min=0), dim=0)
        G3 = torch.sum(torch.clamp(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                    for mean, sd in options], dim=0), min=0), dim=0)
    elif version == 'mean-scaled':
        O1 = torch.clamp(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                          for mean, slp in options], dim=0), min=0)
        # Operation #2, G neurons independently summarizing these inputs with noise
        G1 = torch.sum(torch.clamp(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options], dim=0), min=0), dim=0)
        G2 = torch.sum(torch.clamp(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options], dim=0), min=0), dim=0)
        G3 = torch.sum(torch.clamp(torch.stack([torch.normal(mean=mean, std=(slp * mean) ** 0.5, size=(num_samples,), device=device)
                                    for mean, slp in options], dim=0), min=0), dim=0)
    # Operation #3, every time access to the value of the item is different, even within the same trial and for the same option,
    # e.g, V1 in the numerator and in the denominator of option 1 are independently drawn
    Context = torch.stack((G1, G2, G3), dim=0)
    O3 = [Rmax*DirectValue/(M+ContextValue*w) for DirectValue, ContextValue in zip(O1, Context)]
    # Alternatively, use the shared denominator by assuming noise varying trial-by-trial but fixed within trial
    # D = torch.sum(O1, dim=0)
    # O3 = [Rmax * DirectValue / (M + D * w) for DirectValue in O1]
    # Operation #4, apply late noise
    Outputs = torch.clamp(torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device)
                           for ComputedValue in O3]), min=0)
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
    x = np.arange(SVs.min(), SVs.max(), interval)
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
