import os
from myFunctions import load_data, DN, dDN, dDNb, merge_pdf_files
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join as join

# Switch under the working directory
os.chdir(r'/Users/bs3667/PycharmProjects/NoiseProject')
# Define I/O directories
datadir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/TaskProgram/log/txtDat'
mydatdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/myData'
svdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults'

# load fitting results
# fit = pd.read_csv(r'/Users/bs3667/Noise/modelfit/Results/FastBADS_Mtlb/Rslts_FastBADS_Best.txt', sep='\t')
fit = pd.read_csv(r'/Users/bs3667/Noise/modelfit/Results/GridSearch_Mtlb/Rslts_GrdSrch.txt', sep='\t')
# loading log data
mt = pd.read_csv(join(mydatdir, 'TrnsfrmData.csv'))

# generating choice based on data in mt and
colors = ["red", "purple", "cyan"]
sublist = np.unique(mt.subID)
subj = 0
pdf_files = []
while subj < len(sublist):
    if subj % 12 == 0:
        fig, axs = plt.subplots(4, 3, figsize=(8.5, 11))
        lgd = True
    for r in range(4):
        for c in range(3):
            print(f"#{subj}, {sublist[subj]}")
            pars = fit[(fit['subID']==(subj+1)) & (fit['modeli']==3)]
            indvdat = mt[mt["subID"]==sublist[subj]]
            data = indvdat[['V1', 'V2', 'V3']]
            probs = [DN(row.V1, row.V2, row.V3, pars['Mp'].item(), pars['wp'].item()) for row in
                     data.itertuples()]
            indvdat.loc[:, ['pr1', 'pr2', 'pr3']] = torch.stack(probs).cpu().numpy()
            wide_df = indvdat.groupby(['V1', 'V2'])[['subID', 'ID1', 'ID2', 'pr1', 'pr2', 'pr3']].mean()
            wide_df = wide_df.reset_index()
            grouped = wide_df.melt(id_vars=['subID', 'V1', 'V2', 'ID1', 'ID2'], value_vars=['pr1', 'pr2', 'pr3'],
                                   var_name='Option', value_name='pr')
            sns.scatterplot(x="V1", y="V2", hue="Option", size="pr",
                            sizes=(12, 300), alpha=.45, legend=lgd, palette=colors, data=grouped, ax=axs[r, c])
            if lgd:
                axs[r, c].legend(fontsize='5')
            lgd = False
            axs[r, c].set_title(f'#{subj + 1}, {sublist[subj]}')
            subj += 1
    plt.tight_layout()
    filename = join(svdir, 'Individual_choice', 'tgrtPairs_toSubj' + str(subj) + '.pdf')
    pdf_files.append(filename)
    plt.savefig(filename, format='pdf')
    plt.close(fig)
output_file = join(svdir, 'Individual_choice', 'PstPredctvChk_GrdSrch.pdf')
merge_pdf_files(pdf_files, output_file)
