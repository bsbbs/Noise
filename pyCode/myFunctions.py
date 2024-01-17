import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Distributive divisive normalization, with additive early noise (when version == 'additive') or with mean-scaled noise (version == 'mean-scaled')
# type 1: fully dependent
def dDNDpdnt(V1, V2, V3, eta, version):
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
def dDNwOpt(V1, V2, V3, eta, version):  # dependent within options
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
def dDN(V1, V2, V3, eta, version): # fully independent
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
def dDNb(V1, V2, V3, eta, version): # fully independent and non-negative activities
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
    import scipy.stats as stats
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
    ax.plot(x[mask], pdfs[0][mask], label='Option 1', color='black', linewidth=0.5)
    mask = pdfs[1] > 1e-4
    ax.plot(x[mask], pdfs[1][mask],  label='Option 2', color='black', linewidth=0.5)
    mask = pdfs[2] > 1e-4
    ax.plot(x[mask], pdfs[2][mask],  label='Option 3', color='red', linewidth=0.5, zorder=1)
    overlap = np.minimum(pdfs[0], pdfs[1])
    cutout = np.minimum(overlap, pdfs[2])
    mask = overlap > 1e-4
    ax.fill_between(x[mask], cutout[mask], overlap[mask], color='black', label='Overlap Area')
    ax.fill_between(x[mask], 0, cutout[mask], color='red', alpha=0.5, label='Overlap Area')
    ax.set_xlim(0, 45)
    ax.set_xticks([0, 20, 40])
    ax.set_ylim(0, .5)
    ax.set_yticks([0, .5])
    ax.set_xlabel('Neural activity')
    ax.set_ylabel('Probability density')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(axis='both', direction='out', width=0.5, length=2)
def AUC(pdfs, x):
    interval = x[1] - x[0]
    overlap = np.minimum(pdfs[0], pdfs[1])
    AUCval = sum(overlap) * interval * 100
    return AUCval
