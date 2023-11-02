import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.animation import FuncAnimation
from os.path import join as join
svdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults'

# Animation, change of V1 - V2 overlapping by moving V3
V3mean = np.arange(0, 50, .25)
V1=[50, 5] # mean and std
V2=[58, 5]
eta=0.3
# Model 1 - Linear subtraction
def Model1(V1, V2, V3, eta):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use
    options = torch.tensor([V1, V2, V3])
    num_samples=1000000
    w=0.1
    # Operation #1, sampling with early noise
    O1 = [meanval + torch.normal(mean=0, std=sdval, size=(num_samples,), device=device) for meanval, sdval in
           options]
    # Operation #2, implementing lateral inhibition
    O2 = [DirectValue -torch.sum(torch.stack(O1, dim=0), dim=0)*w for DirectValue in O1]
    # Operation #3, add late noise
    Outputs = [ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device) for ComputedValue in O2]
    return torch.stack(Outputs).numpy()
def update1(frame):
    V3=[V3mean[frame], 5]
    SVs = Model1(V1, V2, V3, eta)
    # pdf3 = stats.norm.pdf(x, np.mean(SVs[2]), np.std(SVs[2]))
    xs = [np.arange(np.mean(SV)-3*np.std(SV), np.mean(SV)+3*np.std(SV), .1) for SV in SVs]
    pdfs = [stats.norm.pdf(x, np.mean(SV), np.std(SV)) for x, SV in zip(xs, SVs)]
    x = np.arange(0, 62, .1)
    pdf1 = stats.norm.pdf(x, np.mean(SVs[0]), np.std(SVs[0]))
    pdf2 = stats.norm.pdf(x, np.mean(SVs[1]), np.std(SVs[1]))
    overlap = np.minimum(pdf1, pdf2)
    plt.clf()
    plt.plot(xs[0], pdfs[0], label='Option 1', color='limegreen')
    plt.plot(xs[1], pdfs[1], label='Option 2', color='steelblue')
    plt.plot(xs[2], pdfs[2], label='Option 3', color='tomato')
    mask=overlap>1e-4
    plt.fill_between(x[mask], 0, overlap[mask], color='darkslategrey', label='Overlap Area')
    peak_option1 = np.argmax(pdfs[0])
    peak_option2 = np.argmax(pdfs[1])
    plt.plot(47.19392652511616, 0.0873392664394644 * 1.1, 'v', color='gray', markersize=6)
    plt.plot(39.19906539917012, 0.0873392664394644 * 1.1, 'v', color='gray', markersize=6)
    plt.plot(xs[0][peak_option1], pdfs[0][peak_option1]*1.1,'vk', markersize=6)
    plt.plot(xs[1][peak_option2], pdfs[0][peak_option1]*1.1,'vk', markersize=6)
    plt.xlim((0, 62))
    plt.ylim((0, .1))
    #plt.title(f'Distractor value ={V3mean[frame]}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
fig = plt.figure(figsize=(4, 2))
# Create the animation
ani = FuncAnimation(fig, update1, frames=len(V3mean), repeat=True, interval=1000/60)
# Save the animation as a GIF
ani.save(join(svdir, 'ModelSimulation', 'Animation_Model1.gif'), writer='pillow')  # You may need to install the 'pillow' package

# Model 2 - Divisive normalization
def Model2(V1, V2, V3, eta):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use
    options = torch.tensor([V1, V2, V3])
    num_samples=1000000
    w=0.1
    Rmax = 750
    # Operation #1, sampling with early noise
    O1 = [meanval + torch.normal(mean=0, std=sdval, size=(num_samples,), device=device) for meanval, sdval in
           options]
    # Operation #2, implementing lateral inhibition
    O2 = [Rmax*DirectValue/torch.sum(torch.stack(O1, dim=0), dim=0)*w for DirectValue in O1]
    # Operation #3, add late noise
    Outputs = [ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device) for ComputedValue in O2]
    return torch.stack(Outputs).numpy()
V1=[50, 0] # mean and std
V2=[58, 0]
eta=2
def update2(frame):
    V3=[V3mean[frame], 0]
    SVs = Model2(V1, V2, V3, eta)
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
    plt.plot(34.817720651626715, 0.1334456413046007 * 1.1, 'v', color='gray', markersize=6)
    plt.plot(40.35710206031813, 0.1334456413046007 * 1.1, 'v', color='gray', markersize=6)
    plt.plot(xs[0][peak_option1], pdfs[0][peak_option1]*1.1,'vk', markersize=6)
    plt.plot(xs[1][peak_option2], pdfs[0][peak_option1]*1.1,'vk', markersize=6)
    plt.xlim((0, 50))
    plt.ylim((0, .23))
    #plt.title(f'Distractor value ={V3mean[frame]}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
fig = plt.figure(figsize=(4, 2))
# Create the animation
ani = FuncAnimation(fig, update2, frames=len(V3mean), repeat=True, interval=1000/60)
# Save the animation as a GIF
ani.save(join(svdir, 'ModelSimulation', 'Animation_Model2Conditional_Late.gif'), writer='pillow')  # You may need to install the 'pillow' package
