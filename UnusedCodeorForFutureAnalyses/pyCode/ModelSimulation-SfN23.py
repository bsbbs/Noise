import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from os.path import join as join
svdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults'

# Animation, change of V1 - V2 overlapping by moving V3
#V3mean = np.arange(0, 50, .25)
#V3mean = np.arange(0, 2400, 12)**0.5#np.arange(0, 19600, 98)**0.5
V1mean = 45 #150
V2mean = 58 #158
color1 = 'limegreen'
color2 = 'steelblue'
color3 = 'tomato'
color1 = 'orange'
color2 = 'red'
color3 = 'green'
xlim = 73
ylim = 55

# Model 2 - Divisive normalization

def DN(V1, V2, V3, eta):
    w = 1
    M = 1
    Rmax = 75
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use
    options = torch.tensor([V1, V2, V3])
    num_samples = int(1e5)
    # Operation #1, sampling with early noise
    O1 = torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                      for mean, sd in options], dim=0)
    # Operation #2, G neurons independently summarizing these inputs with noise
    G1 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                for mean, sd in options], dim=0), dim=0)
    G2 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                for mean, sd in options], dim=0), dim=0)
    G3 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device)
                                for mean, sd in options], dim=0), dim=0)
    Context = torch.stack((G1, G2, G3), dim=0)
    # Operation #3, implementing lateral inhibition
    # D = torch.sum(O1, dim=0)
    # O3 = [Rmax * DirectValue / (M + D * w) for DirectValue in O1]
    O3 = [Rmax*DirectValue/(M+ContextValue*w) for DirectValue, ContextValue in zip(O1, Context)]
    # Operation #4, add late noise
    Outputs = torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device)
                           for ComputedValue in O3])
    max_indices = torch.argmax(Outputs, dim=0)
    max_from_each_distribution = torch.zeros_like(Outputs)
    max_from_each_distribution[max_indices, torch.arange(Outputs.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / Outputs.shape[1]
    return Outputs.numpy(), probs.numpy()

def DNms(V1, V2, V3, eta): # Mean-scaled noise
    w = 1
    M = 1
    Rmax = 75
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use
    options = torch.tensor([V1, V2, V3])
    num_samples = int(1e6)
    slp = 2.52 # the slope for precise items in the data
    # Operation #1, sampling with early noise
    O1 = torch.stack([torch.normal(mean=mean, std=(slp*mean)**0.5, size=(num_samples,), device=device)
                      for mean, slp in options], dim=0)
    # Operation #2, G neurons independently summarizing these inputs with noise
    G1 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp*mean)**0.5, size=(num_samples,), device=device)
                                for mean, slp in options], dim=0), dim=0)
    G2 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp*mean)**0.5, size=(num_samples,), device=device)
                                for mean, slp in options], dim=0), dim=0)
    G3 = torch.sum(torch.stack([torch.normal(mean=mean, std=(slp*mean)**0.5, size=(num_samples,), device=device)
                                for mean, slp in options], dim=0), dim=0)
    Context = torch.stack((G1, G2, G3), dim=0)
    # Operation #3, implementing lateral inhibition
    # D = torch.sum(O1, dim=0)
    # O3 = [Rmax * DirectValue / (M + D * w) for DirectValue in O1]
    O3 = [Rmax*DirectValue/(M+ContextValue*w) for DirectValue, ContextValue in zip(O1, Context)]
    # Operation #4, add late noise
    Outputs = torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device)
                           for ComputedValue in O3])
    max_indices = torch.argmax(Outputs, dim=0)
    max_from_each_distribution = torch.zeros_like(Outputs)
    max_from_each_distribution[max_indices, torch.arange(Outputs.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / Outputs.shape[1]
    return Outputs.numpy(), probs.numpy()


fig = plt.figure(figsize=(3.2, 3))
colors = ['blue', 'red', 'lightblue', 'pink']
eta = 13
slp = 2.52
V1 = [V1mean, slp]  # mean and the slope of mean-scaled variance
V2 = [V2mean, slp]
V3curves = np.arange(0, V1mean, 2)
ratio = []
for V3mean in V3curves:
    V3 = [V3mean, slp]
    SVs, probs = DNms(V1, V2, V3, eta)
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
def AUC(V3):
    SVs, _ = DN(V1, V2, V3, eta)
    interval = 0.1
    x = np.arange(0, ylim, interval)
    kdes = [stats.gaussian_kde(SV) for SV in SVs[:2]]
    pdfs = [kde(x) for kde in kdes]
    overlap = np.minimum(pdfs[0], pdfs[1])
    AUCval = sum(overlap) * interval * 100
    return AUCval
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
# representation curves
V3curves = np.arange(0,41,10)
fig = plt.figure(figsize=(3.2, 3))
colors = makecolors(V3curves)
ax = plt.gca()
set_axis(ax)
for frame in range(len(V3curves)):
    V = np.arange(0, xlim, 0.1)
    R = Rmax*V/(M+V+V2mean+V3curves[frame])
    ax.plot(V, R, c=colors[frame])
    ax.plot([0, V1mean], [R[V == V1mean], R[V == V1mean]], color=colors[frame], linestyle='--', linewidth=1)
    ax.annotate('V3', (V3curves[frame], -4.5), color=colors[frame], fontsize=12,
                ha='center', va='bottom')
    ax.plot([V1mean, V1mean], [0, Rmax * V1mean / (M + V1mean + V2mean + V3curves.min())], color='black',
            linestyle='--', linewidth=1)
    ax.spines['bottom'].set_position('zero')
    ax.set_xticks([V1mean], labels=['V1'], color='black')
    ax.set_xlim(-11, xlim)
    ax.set_ylim(-4.5, ylim)
    ax.set_xlabel('Inputs')
    ax.set_ylabel('Neural activity')  # adjust padding
    plt.tight_layout()
    plt.savefig(join(svdir, 'ModelSimulation', f'Curve_DN_{Test}{frame}.pdf'), format='pdf')

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

# Chosen ratio
V3 = [V3curves.min(), eps]
_, probs = DN(V1, V2, V3, eta)
ratio1 = probs[1]/(probs[0]+probs[1])*100
V3 = [V3curves.max(), eps]
_, probs = DN(V1, V2, V3, eta)
ratio2 = probs[1]/(probs[0]+probs[1])*100
ratiorng = np.array([ratio1, ratio2])
_, ax = plt.subplots(1,3)
fig = plt.figure(figsize=(3.2*3, 3))
ax[0] = fig.add_axes([.03, .1, .35, .9])
ax[1] = fig.add_axes([.465, .2, .2, .6])
ax[2] = fig.add_axes([.78, .2, .2, .6])
def ChosenRatio(frame):
    V3 = [V3curves[frame], eps]
    SVs, probs = DN(V1, V2, V3, eta)
    # chosen ratio
    ratio = probs[1] / (probs[0] + probs[1])*100
    ax[2].plot(V3curves[frame], ratio, '.', label="% Correct (V1 & V2)", color=colors[frame])
    ax[2].set_xlabel('V3')
    ax[2].set_ylabel('% Correct (V1 & V2)')  # adjust padding
    ax[2].set_ylim([ratiorng.min() - 1, ratiorng.max() + 1])
    ax[2].set_xlim([V3curves.min() - 1, V3curves.max() + 1])
    ax[2].set_title('Relative accuracy')
    # AUC
    AUCPlot(ax[1], SVs, V3curves, AUCrng, colors, frame)
    # Distribution
    pdfs, y = getpdfs(SVs)
    ax[0].clear()
    set_axis(ax[0])
    adddistrilines(ax[0], SVs, pdfs, y, V1, V2, V3, maxheight, frame, overlap0)
    if (frame==0) | (frame==len(V3curves)-1):
        plt.savefig(join(svdir, 'ModelSimulation', f'ChosenRatio_DN_{Test}{frame}.pdf'), format='pdf')
ani = FuncAnimation(fig, ChosenRatio, frames=len(V3curves), repeat=False, blit=False)
ani.save(join(svdir, 'ModelSimulation', f'ChosenRatio_DN_{Test}.gif'), writer='pillow', fps=20)

# Mixed noise
Test = 'Mixed'
#eta=3
#eps=5
epsvec = np.linspace(0, 4.4, 8)
etavec = np.linspace(3, 0, 8)
#colors = ('viridis')
gcolors = sns.color_palette('Spectral', 8)
#sns.set_palette('Spectral')
fig = plt.figure(figsize=(3.2, 3))
def MixedRatios(frame):
    eps = epsvec[frame]
    eta = etavec[frame]
    V1=[V1mean, eps] # mean and std
    V2=[V2mean, eps]
    V3curves = np.linspace(0, V1mean-3, 50)
    ratio = []
    for V3mean in V3curves:
        V3 = [V3mean, eps]
        SVs, probs = DN(V1, V2, V3, eta)
        ratio.append(probs[1] / (probs[0] + probs[1])*100)
    plt.scatter(V3curves, ratio, color=gcolors[frame], marker='.', s=30)
    plt.xlim((-3, V3curves.max() + 3))
    #plt.yticks([71, 74, 77, 80])
    plt.xlabel('V3')
    plt.ylabel('% Correct (V1 & V2)')
    plt.title('Relative accuracy')
    plt.tight_layout()
    if frame==0:
        plt.savefig(join(svdir, 'ModelSimulation', f'MixedRatios_DN_{Test}{frame}.pdf'), format='pdf')
ani = FuncAnimation(fig, MixedRatios, frames=len(V3curves), repeat=False)
ani.save(join(svdir, 'ModelSimulation', f'MixedRatios_DN_{Test}.gif'), writer='pillow', fps=20)





def update(frame):
    V3 = [V3mean[frame], eps]
    SVs, _ = DN(V1, V2, V3, eta)
    ys = [np.arange(np.mean(SV) - 3 * np.std(SV), np.mean(SV) + 3 * np.std(SV), .1) for SV in SVs]
    pdfs = [stats.norm.pdf(y, np.mean(SV), np.std(SV)) for y, SV in zip(ys, SVs)]
    plt.clf()
    # replace x and y axis with arrows
    ax = plt.gca()
    set_axis(ax)
    # plot distribution outside of the axes
    V = np.arange(0, xlim, 0.1)
    R = Rmax*V/(M+V+V1mean+V2mean+V3mean[frame])
    ax.plot(V, R, label='Representation', color='black')
    ax.plot(-pdfs[0]/pdfs[0].max()*10, ys[0], label='Option 1', color=color1)
    ax.plot([-10, V1mean], [np.mean(SVs[0]), np.mean(SVs[0])], color=color1, linestyle='--', linewidth=1)
    ax.plot([V1mean, V1mean], [0, np.mean(SVs[0])], color=color1, linestyle='--', linewidth=1)
    ax.plot(-pdfs[1]/pdfs[0].max()*10, ys[1], label='Option 2', color=color2)
    ax.plot([-10, V2mean], [np.mean(SVs[1]), np.mean(SVs[1])], color=color2, linestyle='--', linewidth=1)
    ax.plot([V2mean, V2mean], [0, np.mean(SVs[1])], color=color2, linestyle='--', linewidth=1)
    ax.plot(-pdfs[2]/pdfs[0].max()*10, ys[2], label='Option 3', color=color3)
    # ax.plot([-10, V3mean[frame]], [np.mean(SVs[2]), np.mean(SVs[2])], color=color3, linestyle='--')
    # ax.plot([V3mean[frame], V3mean[frame]], [0, np.mean(SVs[2])], color=color3, linestyle='--')
    if V1[1] > 0:
        xs = [np.arange(meanval - 3 * sdval, meanval + 3 * sdval, .1) for meanval, sdval in [V1, V2, V3]]
        pdfxs = [stats.norm.pdf(np.arange(meanval - 3 * sdval, meanval + 3 * sdval, .1), meanval, sdval) for meanval, sdval in [V1, V2, V3]]
        ax.plot(xs[0], pdfxs[0] / pdfxs[0].max() * 5, label='V1', color=color1)
        ax.plot(xs[1], pdfxs[1] / pdfxs[0].max() * 5, label='V2', color=color2)
        ax.plot(xs[2], pdfxs[2] / pdfxs[0].max() * 5, label='V3', color=color3)
    ax.set_xticks([V3mean[frame], V1mean, V2mean])
    ax.set_xticklabels(['V3', 'V1', 'V2'])
    # Set x and y-axis limits
    ax.set_xlim(-11, xlim)
    ax.set_ylim(0, ylim)
    ax.spines['left'].set_position('zero')
    ax.set_xlabel('Inputs')
    ax.yaxis.set_label_position('left')  # Position the y-label on the left side
    ax.set_ylabel('Neural activity', labelpad=-20)  # adjust padding
    plt.tight_layout()
# Create a figure and axis
fig = plt.figure(figsize=(3.4, 3))
#fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=len(V3mean), repeat=False, blit=False)
ani.save(join(svdir, 'ModelSimulation', 'GIF_DN_Late.gif'), writer='pillow')
ani.save(join(svdir, 'ModelSimulation', 'Movie_DN_Late.mp4'), writer='ffmpeg')


def update2(frame):
    V3=[V3mean[frame], eps]
    SVs, _ = DN(V1, V2, V3, eta)
    xs = [np.arange(np.mean(SV)-3*np.std(SV), np.mean(SV)+3*np.std(SV), .1) for SV in SVs]
    pdfs = [stats.norm.pdf(x, np.mean(SV), np.std(SV)) for x, SV in zip(xs, SVs)]
    x = np.arange(0, 50, .1)
    pdf1 = stats.norm.pdf(x, np.mean(SVs[0]), np.std(SVs[0]))
    pdf2 = stats.norm.pdf(x, np.mean(SVs[1]), np.std(SVs[1]))
    pdf3 = stats.norm.pdf(x, np.mean(SVs[2]), np.std(SVs[2]))
    overlap = np.minimum(pdf1, pdf2)
    cutout = np.minimum(overlap, pdf3)
    plt.clf()
    plt.plot(xs[0], pdfs[0], label='Option 1', color='limegreen')
    plt.plot(xs[1], pdfs[1], label='Option 2', color='steelblue')
    plt.plot(xs[2], pdfs[2], label='Option 3', color='tomato')
    mask=overlap>1e-4
    plt.fill_between(x[mask], cutout[mask], overlap[mask], color='darkslategrey', label='Overlap Area')
    plt.fill_between(x[mask], 0, cutout[mask], color='darkslategrey', alpha=0.5, label='Overlap Area')
    peak_option1 = np.argmax(pdfs[0])
    peak_option2 = np.argmax(pdfs[1])
    plt.plot(36.408478832244946, 0.25818047222166446, 'v', color='gray', markersize=6, alpha=0.5)
    plt.plot(38.356777167320324, 0.25818047222166446, 'v', color='gray', markersize=6, alpha=0.5)
    plt.plot(xs[0][peak_option1], pdfs[0][peak_option1]*1.1,'vk', markersize=6, alpha=0.5)
    plt.plot(xs[1][peak_option2], pdfs[0][peak_option1]*1.1,'vk', markersize=6, alpha=0.5)
    plt.xlim((0, 45))
    plt.ylim((0, .4))
    plt.xlabel('Neural activity')
    plt.ylabel('Density')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
fig = plt.figure(figsize=(4, 2))
ani = FuncAnimation(fig, update2, frames=len(V3mean), repeat=False, interval=1000/60)
ani.save(join(svdir, 'ModelSimulation', 'Movie_DN_Late.gif'), writer='pillow')

V3meanp = np.arange(0, V1mean, 1)
ratio = []
for frame in range(len(V3meanp)):
    V3=[V3meanp[frame], eps]
    _, probs = DN(V1, V2, V3, eta)
    ratio.append(probs[1]/(probs[0]+probs[1]))
accuracy = np.array(ratio)*100
fig = plt.figure(figsize=(2.4, 2))
cmap = plt.get_cmap('viridis')
plt.scatter(V3meanp, accuracy, c=accuracy, cmap=cmap, marker='.', s=30)
plt.xlim((-3, V3meanp.max()+3))
plt.yticks([71,74,77,80])
plt.xlabel('V3')
plt.ylabel('% correct | 1,2')
plt.tight_layout()
plt.show()
plt.savefig(join(svdir, 'ModelSimulation', 'Probs_DN_Late.pdf'), format='pdf')

# Early noise only
eps=6
eta=0
V1=[V1mean, eps] # mean and std
V2=[V2mean, eps]
def update2(frame):
    V3=[V3mean[frame], eps]
    SVs, _ = DN(V1, V2, V3, eta)
    # pdf3 = stats.norm.pdf(x, np.mean(SVs[2]), np.std(SVs[2]))
    xs = [np.arange(np.mean(SV)-3*np.std(SV), np.mean(SV)+3*np.std(SV), .1) for SV in SVs]
    pdfs = [stats.norm.pdf(x, np.mean(SV), np.std(SV)) for x, SV in zip(xs, SVs)]
    x = np.arange(0, 50, .1)
    pdf1 = stats.norm.pdf(x, np.mean(SVs[0]), np.std(SVs[0]))
    pdf2 = stats.norm.pdf(x, np.mean(SVs[1]), np.std(SVs[1]))
    pdf3 = stats.norm.pdf(x, np.mean(SVs[2]), np.std(SVs[2]))
    overlap = np.minimum(pdf1, pdf2)
    cutout = np.minimum(overlap, pdf3)
    plt.clf()
    plt.plot(xs[0], pdfs[0], label='Option 1', color='limegreen')
    plt.plot(xs[1], pdfs[1], label='Option 2', color='steelblue')
    plt.plot(xs[2], pdfs[2], label='Option 3', color='tomato')
    mask=overlap>1e-4
    plt.fill_between(x[mask], cutout[mask], overlap[mask], color='darkslategrey', label='Overlap Area')
    plt.fill_between(x[mask], 0, cutout[mask], color='darkslategrey', alpha=0.5, label='Overlap Area')
    peak_option1 = np.argmax(pdfs[0])
    peak_option2 = np.argmax(pdfs[1])
    plt.plot(36.42475504875191, 0.22995411540267083, 'v', color='gray', markersize=6, alpha=0.5)
    plt.plot(38.438123011589134, 0.22995411540267083, 'v', color='gray', markersize=6, alpha=0.5)
    plt.plot(xs[0][peak_option1], pdfs[0][peak_option1]*1.1,'vk', markersize=6, alpha=0.5)
    plt.plot(xs[1][peak_option2], pdfs[0][peak_option1]*1.1,'vk', markersize=6, alpha=0.5)
    plt.xlim((0, 45))
    plt.ylim((0, .4))
    #plt.title(f'Distractor value ={V3mean[frame]}')
    plt.xlabel('Neural activity')
    plt.ylabel('Density')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
fig = plt.figure(figsize=(4, 2))
# Create the animation
ani = FuncAnimation(fig, update2, frames=len(V3mean), repeat=False, interval=1000/60)
# Save the animation as a GIF
ani.save(join(svdir, 'ModelSimulation', 'Movie_DN_Early.gif'), writer='pillow')  # You may need to install the 'pillow' package

V3meanp = np.arange(0, V1mean, 1)
ratio = []
for frame in range(len(V3meanp)):
    V3=[V3meanp[frame], eps]
    _, probs = DN(V1, V2, V3, eta)
    ratio.append(probs[1]/(probs[0]+probs[1]))
accuracy = np.array(ratio)*100
fig = plt.figure(figsize=(2.4, 2))
cmap = plt.get_cmap('viridis')
plt.scatter(V3meanp, accuracy, c=accuracy, cmap=cmap, marker='.', s=30)
plt.xlim((-3, V3meanp.max()+3))
plt.yticks([76,78,80])
plt.xlabel('V3')
plt.ylabel('% correct | 1,2')
plt.tight_layout()
plt.show()
plt.savefig(join(svdir, 'ModelSimulation', 'Probs_DN_Early.pdf'), format='pdf')

