import os
import glob
import pandas as pd
def load_data(directory, file_pattern):
    # Directory path
    # directory = r"C:\Users\Bo\Dropbox (NYU Langone Health)\CESS-Bo\TaskProgram\log\txtDat"

    # Pattern for file names
    #file_pattern = "BidTask_22*"

    # List to store all file paths
    file_paths = glob.glob(os.path.join(directory, file_pattern))

    # List to store data from all files
    data = []

    # Iterate through each file
    for file_path in file_paths:
        # Read the file data
        with open(file_path, 'r') as file:
            file_data = pd.read_csv(file, sep='\t')
            # Process the file data if needed
            # ...
            # Append the processed data to the list
            data.append(file_data)

    # Combine all DataFrames into a single DataFrame
    df = pd.concat(data, ignore_index=True)

    # Display the DataFrame
    return df

# The simplest model, we call that McFadden model, with only decision noise.
# Value of the option set as the mean bid value
def McFadden(mean1, mean2, mean3, eta):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use

    show = False
    if show:
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt
    options = torch.tensor([(mean1, eta/(2**.5)), (mean2, eta/(2**.5)), (mean3, eta/(2**.5))])
    # sampling with decision noise based on the mean bid value of each option
    num_samples = 100000
    SVs = [meanval + torch.normal(mean=0, std=sdval, size=(num_samples,), device=device) for meanval, sdval in
           options]  # move to device

    if show:
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 5))
        sns.set_palette('Paired')
        palette = sns.color_palette('Paired')
        # Convert tensors back to CPU for plotting
        axs[0].axvline(x=mean1, color=palette[1], linestyle='-', label='Option 1')
        axs[0].axvline(x=mean2, color=palette[2], linestyle='-', label='Option 2')
        axs[0].axvline(x=mean3, color=palette[3], linestyle='-', label='Option 3')
        axs[0].set_xlim(0, max(mean1, mean2, mean3) * 1.2)
        axs[0].set_xlabel('Mean bid value')
        axs[0].set_ylabel('')
        axs[0].legend()
        bins = np.linspace(min(SVs[0].min(), SVs[1].min(), SVs[2].min()).cpu(),
                           max(SVs[0].max(), SVs[1].max(), SVs[2].max()).cpu(), 100)
        sns.histplot(SVs[0].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 1', ax=axs[1])
        sns.histplot(SVs[1].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 2', ax=axs[1])
        sns.histplot(SVs[2].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 3', ax=axs[1])
        axs[1].set_xlabel('Mean bid value with decision noise')
        axs[1].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('McFadden_demo.pdf', format='pdf')

    SVs_tensor = torch.stack(SVs)  # Combine the samples into a single tensor
    # Find the index of the distribution with the max value in each row
    max_indices = torch.argmax(SVs_tensor, dim=0)
    # Create a tensor where each row is 1 if the max value in that row is from the corresponding distribution, and 0 otherwise
    max_from_each_distribution = torch.zeros_like(SVs_tensor)
    max_from_each_distribution[max_indices, torch.arange(SVs_tensor.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / SVs_tensor.shape[1]
    # Print the probabilities
    if show:
        print(f"Probability that a variable drawn from sample 1 is: {probs[0]}")
        print(f"Probability that a variable drawn from sample 2 is: {probs[1]}")
        print(f"Probability that a variable drawn from sample 3 is: {probs[2]}")

    return probs

# Model without normalization, but with representation noise
# This model has no name yet, just call it model 2
def Mdl2(mean1, mean2, mean3, sd1, sd2, sd3, eta):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use

    show = False
    if show:
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt

    num_samples = 100000
    options = ([(mean1, sd1), (mean2, sd2), (mean3, sd3)])
    samples = [torch.normal(mean=mean, std=sd, size=(num_samples,), device=device) for mean, sd in
               options]  # move to device
    # decision noise for each option
    SVs = [sample + torch.normal(mean=0, std=eta, size=(num_samples,), device=device) for sample in
           samples]  # move to device

    if show:
        bins = np.linspace(min(0, min(mean1 - 3 * sd1, mean2 - 3 * sd2, mean3 - 3 * sd3)),
                           max(mean1 + 3 * sd1, mean2 + 3 * sd2, mean3 + 3 * sd3), 100)
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 5))
        sns.set_palette('Paired')
        # Convert tensors back to CPU for plotting
        sns.histplot(samples[0].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 1', ax=axs[0])
        sns.histplot(samples[1].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 2', ax=axs[0])
        sns.histplot(samples[2].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 3', ax=axs[0])
        axs[0].set_xlabel('Bid value with representation noise')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        bins = np.linspace(min(SVs[0].min(), SVs[1].min(), SVs[2].min()).cpu(),
                           max(SVs[0].max(), SVs[1].max(), SVs[2].max()).cpu(), 100)
        sns.histplot(SVs[0].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 1', ax=axs[1])
        sns.histplot(SVs[1].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 2', ax=axs[1])
        sns.histplot(SVs[2].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 3', ax=axs[1])
        axs[1].set_xlabel('Bid value with decision noise')
        axs[1].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('Mdl2_demo.pdf', format='pdf')

    SVs_tensor = torch.stack(SVs)  # Combine the samples into a single tensor
    # Find the index of the distribution with the max value in each row
    max_indices = torch.argmax(SVs_tensor, dim=0)
    # Create a tensor where each row is 1 if the max value in that row is from the corresponding distribution, and 0 otherwise
    max_from_each_distribution = torch.zeros_like(SVs_tensor)
    max_from_each_distribution[max_indices, torch.arange(SVs_tensor.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / SVs_tensor.shape[1]
    # Print the probabilities
    if show:
        print(f"Probability that a variable drawn from sample 1 is: {probs[0]}")
        print(f"Probability that a variable drawn from sample 2 is: {probs[1]}")
        print(f"Probability that a variable drawn from sample 3 is: {probs[2]}")

    return probs


# Canonical Divisive Normalization, with only decision noise
def DN(mean1, mean2, mean3, Mp, wp):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use

    show = False
    if show:
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt
    # the value of denominator
    D = Mp + wp*(mean1 + mean2 + mean3)
    # decision noise for each option. This means the pooled sd from two options is z-scored as 1
    sd = 2**-.5
    options = torch.tensor([(mean1, sd), (mean2, sd), (mean3, sd)])
    # sampling with decision noise based on the normalized value of each option
    num_samples = 100000
    SVs = [torch.normal(mean=meanval / D, std=sdval, size=(num_samples,), device=device) for meanval, sdval in options]  # move to device

    if show:
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 5))
        sns.set_palette('Paired')
        palette = sns.color_palette('Paired')
        # Convert tensors back to CPU for plotting
        axs[0].axvline(x=mean1, color=palette[1], linestyle='-', label='Option 1')
        axs[0].axvline(x=mean2, color=palette[2], linestyle='-', label='Option 2')
        axs[0].axvline(x=mean3, color=palette[3], linestyle='-', label='Option 3')
        axs[0].set_xlim(-0.1, max(mean1, mean2, mean3)*1.2)
        axs[0].set_xlabel('Input value')
        axs[0].set_ylabel('')
        axs[0].legend()
        bins = np.linspace(min(SVs[0].min(),SVs[1].min(),SVs[2].min()).cpu(), max(SVs[0].max(),SVs[1].max(),SVs[2].max()).cpu(), 100)
        sns.histplot(SVs[0].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 1', ax=axs[1])
        sns.histplot(SVs[1].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 2', ax=axs[1])
        sns.histplot(SVs[2].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 3', ax=axs[1])
        axs[1].set_xlabel('Divisively normalized value with noise')
        axs[1].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('DN_demo.pdf', format='pdf')

    SVs_tensor = torch.stack(SVs)  # Combine the samples into a single tensor
    # Find the index of the distribution with the max value in each row
    max_indices = torch.argmax(SVs_tensor, dim=0)
    # Create a tensor where each row is 1 if the max value in that row is from the corresponding distribution, and 0 otherwise
    max_from_each_distribution = torch.zeros_like(SVs_tensor)
    max_from_each_distribution[max_indices, torch.arange(SVs_tensor.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / SVs_tensor.shape[1]
    # Print the probabilities
    if show:
        print(f"Probability that a variable drawn from sample 1 is: {probs[0]}")
        print(f"Probability that a variable drawn from sample 2 is: {probs[1]}")
        print(f"Probability that a variable drawn from sample 3 is: {probs[2]}")

    return probs

# Distributional Divisive Normalization
def dDN(mean1, mean2, mean3, sd1, sd2, sd3, Mp, wp, mode):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use

    show = False
    if show:
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt

    num_samples = 100000
    options = torch.tensor([(mean1, sd1), (mean2, sd2), (mean3, sd3)])
    samples = [torch.normal(mean=mean, std=sd, size=(num_samples,), device=device) for mean, sd in options]  # move to device

    if mode == 'cutoff':
        samples = [sample[sample >= 0] for sample in samples]
        min_length = min([len(sample) for sample in samples])
        samples = [sample[:min_length] for sample in samples]
    elif mode == 'absorb':
        samples = [torch.clamp(sample, min=0) for sample in samples]

    samples_stacked = torch.stack(samples, dim=0)
    D = torch.sum(samples_stacked, dim=0)*wp + torch.tensor(Mp, device=device)
    # decision noise for each option. This means the pooled sd from two options is z-scored as 1
    sd = 2**-.5
    SVs = [sample / D + torch.normal(mean=0, std=sd, size=(len(sample),), device=device) for sample in samples]  # move to device
    if show:
        bins = np.linspace(min(0, min(mean1-3*sd1, mean2-3*sd2, mean3-3*sd3)), max(mean1+3*sd1, mean2+3*sd2, mean3+3*sd3), 100)
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 5))
        sns.set_palette('Paired')
        # Convert tensors back to CPU for plotting
        sns.histplot(samples[0].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 1', ax=axs[0])
        sns.histplot(samples[1].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 2', ax=axs[0])
        sns.histplot(samples[2].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 3', ax=axs[0])
        axs[0].set_xlabel('Input value with '+mode+' at zero')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        bins = np.linspace(min(SVs[0].min(), SVs[1].min(), SVs[2].min()).cpu(),
                           max(SVs[0].max(), SVs[1].max(), SVs[2].max()).cpu(), 100)
        sns.histplot(SVs[0].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 1', ax=axs[1])
        sns.histplot(SVs[1].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 2', ax=axs[1])
        sns.histplot(SVs[2].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 3', ax=axs[1])
        axs[1].set_xlabel('Divisively normalized value with noise')
        axs[1].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('dDN_'+mode+'_demo.pdf', format='pdf')

    SVs_tensor = torch.stack(SVs)  # Combine the samples into a single tensor
    # Find the index of the distribution with the max value in each row
    max_indices = torch.argmax(SVs_tensor, dim=0)
    # Create a tensor where each row is 1 if the max value in that row is from the corresponding distribution, and 0 otherwise
    max_from_each_distribution = torch.zeros_like(SVs_tensor)
    max_from_each_distribution[max_indices, torch.arange(SVs_tensor.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / SVs_tensor.shape[1]
    # Print the probabilities
    if show:
        print(f"Probability that a variable drawn from sample 1 is: {probs[0]}")
        print(f"Probability that a variable drawn from sample 2 is: {probs[1]}")
        print(f"Probability that a variable drawn from sample 3 is: {probs[2]}")

    return probs

# Distributional Divisive Normalization, with the denominator values independent from the numerators
def dDNb(mean1, mean2, mean3, sd1, sd2, sd3, Mp, wp, mode):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # determine the device to use

    show = False
    if show:
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt

    num_samples = 100000
    options = torch.tensor([(mean1, sd1), (mean2, sd2), (mean3, sd3)])
    samples = [torch.normal(mean=mean, std=sd, size=(num_samples,), device=device) for mean, sd in options]  # move to device
    samplesD = [torch.normal(mean=mean, std=sd, size=(num_samples,), device=device) for mean, sd in
               options]  # move to device
    samplesD_stacked = torch.stack(samplesD, dim=0)
    SIGMA = torch.sum(samplesD_stacked, dim=0)
    if mode == 'cutoff':
        SIGMA = SIGMA[SIGMA > 0]
        min_length = len(SIGMA)
        samples = [sample[:min_length] for sample in samples]
    elif mode == 'absorb':
        SIGMA = torch.clamp(SIGMA, min=0)
    D = SIGMA*wp + torch.tensor(Mp, device=device)
    # decision noise for each option. This means the pooled sd from two options is z-scored as 1
    sd = 2**-.5
    SVs = [sample / D + torch.normal(mean=0, std=sd, size=(len(sample),), device=device) for sample in samples]  # move to device
    if show:
        bins = np.linspace(min(0, min(mean1-3*sd1, mean2-3*sd2, mean3-3*sd3)), max(mean1+3*sd1, mean2+3*sd2, mean3+3*sd3), 100)
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 5))
        sns.set_palette('Paired')
        # Convert tensors back to CPU for plotting
        sns.histplot(samples[0].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 1', ax=axs[0])
        sns.histplot(samples[1].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 2', ax=axs[0])
        sns.histplot(samples[2].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 3', ax=axs[0])
        sns.histplot(SIGMA.cpu(), bins=bins, kde=True, alpha=0.5, label='SIGMA', ax=axs[0])
        axs[0].set_xlabel('Input value with '+mode+' at zero')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        bins = np.linspace(min(SVs[0].min(), SVs[1].min(), SVs[2].min()).cpu(),
                           max(SVs[0].max(), SVs[1].max(), SVs[2].max()).cpu(), 100)
        sns.histplot(SVs[0].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 1', ax=axs[1])
        sns.histplot(SVs[1].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 2', ax=axs[1])
        sns.histplot(SVs[2].cpu(), bins=bins, kde=True, alpha=0.5, label='Option 3', ax=axs[1])
        axs[1].set_xlabel('Divisively normalized value with noise')
        axs[1].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('dDNb_'+mode+'_demo.pdf', format='pdf')

    SVs_tensor = torch.stack(SVs)  # Combine the samples into a single tensor
    # Find the index of the distribution with the max value in each row
    max_indices = torch.argmax(SVs_tensor, dim=0)
    # Create a tensor where each row is 1 if the max value in that row is from the corresponding distribution, and 0 otherwise
    max_from_each_distribution = torch.zeros_like(SVs_tensor)
    max_from_each_distribution[max_indices, torch.arange(SVs_tensor.shape[1])] = 1
    probs = torch.sum(max_from_each_distribution, dim=1) / SVs_tensor.shape[1]
    # Print the probabilities
    if show:
        print(f"Probability that a variable drawn from sample 1 is: {probs[0]}")
        print(f"Probability that a variable drawn from sample 2 is: {probs[1]}")
        print(f"Probability that a variable drawn from sample 3 is: {probs[2]}")

    return probs

import re
def reduce_word(word):
    pattern = re.compile('[aeiouAEIOU]')
    reduced_word = re.sub(pattern, '', word)
    return reduced_word

import PyPDF2, os
def merge_pdf_files(pdf_files, output_file):
    # Create a PDF writer object
    pdf_writer = PyPDF2.PdfWriter()

    # Iterate through the PDF files
    for file in pdf_files:
        # Open each PDF file in read-binary mode
        with open(file, 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Iterate through each page of the PDF
            for page_num in range(len(pdf_reader.pages)):
                # Get the page object
                page = pdf_reader.pages[page_num]

                # Add the page to the PDF writer
                pdf_writer.add_page(page)

    # Write the combined PDF to the output file
    with open(output_file, 'wb') as output:
        pdf_writer.write(output)

    print(f"PDF files merged successfully. Merged document saved as '{output_file}'")
    for file in pdf_files:
        os.remove(file)

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
