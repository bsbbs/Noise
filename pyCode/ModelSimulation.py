import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.animation import FuncAnimation
from os.path import join as join
svdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults'

# Animation, change of V1 - V2 overlapping by moving V3
#V3mean = np.arange(0, 50, .25)
V3mean = np.arange(0, 19600, 98)**0.5
V1mean = 150
V2mean = 158


# Model 1 - Linear subtraction
def LS(V1, V2, V3, eta):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use
    options = torch.tensor([V1, V2, V3])
    num_samples=1000000
    w=0.16
    # Operation #1, sampling with early noise
    O1 = [meanval + torch.normal(mean=0, std=sdval, size=(num_samples,), device=device) for meanval, sdval in
           options]
    # Operation #2, Inhibitory neurons summarize independently from inputs
    G1 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device) for mean, sd in
                                options], dim=0), dim=0)
    G2 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device) for mean, sd in
                                options], dim=0), dim=0)
    G3 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device) for mean, sd in
                                options], dim=0), dim=0)
    Context = torch.stack((G1, G2, G3), dim=0)
    # Operation #3, implementing lateral inhibition
    O3 = [DirectValue - ContextValue*w for DirectValue, ContextValue in zip(O1, Context)]
    # Operation #4, add late noise
    Outputs = torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device) for ComputedValue in O3])
    max_indices = torch.argmax(Outputs, dim=0)
    max_from_each_distribution = torch.zeros_like(Outputs)
    max_from_each_distribution[max_indices, torch.arange(Outputs.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / Outputs.shape[1]
    return Outputs.numpy(), probs.numpy()
# Late noise only
eps=0
eta=7/3
V1=[V1mean/3, eps] # mean and std
V2=[V2mean/3, eps]
def update1(frame):
    V3=[V3mean[frame]/3, eps]
    SVs, _ = LS(V1, V2, V3, eta)
    xs = [np.arange(np.mean(SV) - 3 * np.std(SV), np.mean(SV) + 3 * np.std(SV), .1) for SV in SVs]
    pdfs = [stats.norm.pdf(x, np.mean(SV), np.std(SV)) for x, SV in zip(xs, SVs)]
    x = np.arange(0, 45, .1)
    pdf1 = stats.norm.pdf(x, np.mean(SVs[0]), np.std(SVs[0]))
    pdf2 = stats.norm.pdf(x, np.mean(SVs[1]), np.std(SVs[1]))
    pdf3 = stats.norm.pdf(x, np.mean(SVs[2]), np.std(SVs[2]))
    overlap = np.minimum(pdf1, pdf2)
    cutout = np.minimum(overlap, pdf3)
    plt.clf()
    plt.plot(xs[0], pdfs[0], label='Option 1', color='limegreen')
    plt.plot(xs[1], pdfs[1], label='Option 2', color='steelblue')
    plt.plot(xs[2], pdfs[2], label='Option 3', color='tomato')
    mask = overlap > 1e-4
    plt.fill_between(x[mask], cutout[mask], overlap[mask], color='darkslategrey', label='Overlap Area')
    plt.fill_between(x[mask], 0, cutout[mask], color='darkslategrey', alpha=0.5, label='Overlap Area')
    peak_option1 = np.argmax(pdfs[0])
    peak_option2 = np.argmax(pdfs[1])
    plt.plot(33.56813454628001, 0.18792480373328047, 'v', color='gray', markersize=6, alpha=0.5)
    plt.plot(36.231868505478005, 0.18792480373328047, 'v', color='gray', markersize=6, alpha=0.5)
    plt.plot(xs[0][peak_option1], pdfs[0][peak_option1] * 1.1, 'vk', markersize=6, alpha=0.5)
    plt.plot(xs[1][peak_option2], pdfs[0][peak_option1] * 1.1, 'vk', markersize=6, alpha=0.5)
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
ani = FuncAnimation(fig, update1, frames=len(V3mean), repeat=False, interval=1000/60)
ani.save(join(svdir, 'ModelSimulation', 'Movie_LS_Late.gif'), writer='pillow')  # You may need to install the 'pillow' package

V3meanp = np.arange(0, V1mean/3, 1/3)
ratio = []
for frame in range(len(V3meanp)):
    V3=[V3meanp[frame], eps]
    _, probs = LS(V1, V2, V3, eta)
    ratio.append(probs[1]/(probs[0]+probs[1]))
accuracy = np.array(ratio)*100
fig = plt.figure(figsize=(2.4, 2))
cmap = plt.get_cmap('viridis')
plt.scatter(V3meanp, accuracy, c=accuracy, cmap=cmap, marker='.', s=30)
plt.xlim((-1, V3meanp.max()+1))
plt.yticks([79,80,81])
plt.ylim((78.5, 81))
plt.xlabel('V3')
plt.ylabel('% correct | 1,2')
plt.tight_layout()
plt.show()
plt.savefig(join(svdir, 'ModelSimulation', 'Probs_LS_Late.pdf'), format='pdf')

# Early noise only
eps=2
eta=0
V1=[V1mean/3, eps] # mean and std
V2=[V2mean/3, eps]
def update1(frame):
    V3=[V3mean[frame]/3, eps]
    SVs, _ = LS(V1, V2, V3, eta)
    xs = [np.arange(np.mean(SV) - 3 * np.std(SV), np.mean(SV) + 3 * np.std(SV), .1) for SV in SVs]
    pdfs = [stats.norm.pdf(x, np.mean(SV), np.std(SV)) for x, SV in zip(xs, SVs)]
    x = np.arange(0, 45, .1)
    pdf1 = stats.norm.pdf(x, np.mean(SVs[0]), np.std(SVs[0]))
    pdf2 = stats.norm.pdf(x, np.mean(SVs[1]), np.std(SVs[1]))
    pdf3 = stats.norm.pdf(x, np.mean(SVs[2]), np.std(SVs[2]))
    overlap = np.minimum(pdf1, pdf2)
    cutout = np.minimum(overlap, pdf3)
    plt.clf()
    plt.plot(xs[0], pdfs[0], label='Option 1', color='limegreen')
    plt.plot(xs[1], pdfs[1], label='Option 2', color='steelblue')
    plt.plot(xs[2], pdfs[2], label='Option 3', color='tomato')
    mask = overlap > 1e-4
    plt.fill_between(x[mask], cutout[mask], overlap[mask], color='darkslategrey', label='Overlap Area')
    plt.fill_between(x[mask], 0, cutout[mask], color='darkslategrey', alpha=0.5, label='Overlap Area')
    peak_option1 = np.argmax(pdfs[0])
    peak_option2 = np.argmax(pdfs[1])
    plt.plot(33.54848046302804, 0.21147755674321925, 'v', color='gray', markersize=6, alpha=0.5)
    plt.plot(36.21076626777658, 0.21147755674321925, 'v', color='gray', markersize=6, alpha=0.5)
    plt.plot(xs[0][peak_option1], pdfs[0][peak_option1]*1.1, 'vk', markersize=6, alpha=0.5)
    plt.plot(xs[1][peak_option2], pdfs[0][peak_option1]*1.1, 'vk', markersize=6, alpha=0.5)
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
ani = FuncAnimation(fig, update1, frames=len(V3mean), repeat=False, interval=1000/60)
ani.save(join(svdir, 'ModelSimulation', 'Movie_LS_Early.gif'), writer='pillow')  # You may need to install the 'pillow' package

V3meanp = np.arange(0, V1mean/3, 1/3)
ratio = []
for frame in range(len(V3meanp)):
    V3=[V3meanp[frame], eps]
    _, probs = LS(V1, V2, V3, eta)
    ratio.append(probs[1]/(probs[0]+probs[1]))
accuracy = np.array(ratio)*100
fig = plt.figure(figsize=(2.4, 2))
cmap = plt.get_cmap('viridis')
plt.scatter(V3meanp, accuracy, c=accuracy, cmap=cmap, marker='.', s=30)
plt.xlim((-1, V3meanp.max()+1))
plt.yticks([82,83])
plt.ylim((81.5, 83.5))
plt.xlabel('V3')
plt.ylabel('% correct | 1,2')
plt.tight_layout()
plt.show()
plt.savefig(join(svdir, 'ModelSimulation', 'Probs_LS_Early.pdf'), format='pdf')

# Model 2 - Divisive normalization
def DN(V1, V2, V3, eta):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use
    options = torch.tensor([V1, V2, V3])
    num_samples=1000000
    w=1
    M=1
    Rmax = 75
    # Operation #1, sampling with early noise
    O1 = [meanval + torch.normal(mean=0, std=sdval, size=(num_samples,), device=device) for meanval, sdval in
           options]
    # Operation #2, G neurons independently summarizing these inputs with noise
    G1 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device) for mean, sd in
                options], dim=0), dim=0)
    G2 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device) for mean, sd in
                                options], dim=0), dim=0)
    G3 = torch.sum(torch.stack([torch.normal(mean=mean, std=sd, size=(num_samples,), device=device) for mean, sd in
                                options], dim=0), dim=0)
    Context = torch.stack((G1, G2, G3), dim=0)
    # Operation #3, implementing lateral inhibition
    O3 = [Rmax*DirectValue/(M+ContextValue*w) for DirectValue, ContextValue in zip(O1, Context)]
    # Operation #4, add late noise
    Outputs = torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device) for ComputedValue in O3])
    max_indices = torch.argmax(Outputs, dim=0)
    max_from_each_distribution = torch.zeros_like(Outputs)
    max_from_each_distribution[max_indices, torch.arange(Outputs.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / Outputs.shape[1]
    return Outputs.numpy(), probs.numpy()
# Late noise only
eps=0
eta=1.7
V1=[V1mean, eps] # mean and std
V2=[V2mean, eps]
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

