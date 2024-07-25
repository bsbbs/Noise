
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
    Outputs = torch.stack([ComputedValue + torch.normal(mean=0, std=eta, size=(num_samples,), device=device)
                           for ComputedValue in O3])
    max_indices = torch.argmax(Outputs, dim=0)
    max_from_each_distribution = torch.zeros_like(Outputs)
    max_from_each_distribution[max_indices, torch.arange(Outputs.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / Outputs.shape[1]
    return Outputs.numpy(), probs.numpy()
# Late noise only
eps=0
eta=7
V1=[V1mean, eps] # mean and std
V2=[V2mean, eps]
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

V3meanp = np.arange(0, V1mean+7, 1)
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
#plt.yticks([79,80,81])
#plt.ylim((78.5, 81))
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



import numpy as np
import matplotlib.pyplot as plt

# Define the function
def calculate_output(Scale, beta, Cohr, Cmax):
    output_values = Scale * (1 + (1 - np.exp(-beta * Cohr)) * Cmax)
    return output_values

# Define input parameters

beta = 20
Cmax = 1.0
# fig = plt.figure()
fig = plt.figure(figsize=(8, 6))

for Cmax in np.linspace(0.5,1,2):
    for beta in np.linspace(0.5,10,20):
        Cohr = np.linspace(0,1,100)
        Scale = 1.0
        # Calculate the corresponding output values
        output_values = calculate_output(Scale, beta, Cohr, Cmax)

        # Create a plot to visualize the input-output relationship
        plt.plot(Cohr, output_values, label='Output')
        plt.xlabel('c'' values')
        plt.ylabel('V')
        plt.title('Input-Output Relationship of the Function')
        plt.tight_layout()
plt.show()

mean = np.linspace(0,80,80)
sd = (-4/405*(mean**2) + 8/9*mean)
plt.figure()
plt.plot(mean, sd)