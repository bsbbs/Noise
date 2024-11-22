import os
from myFunctions import load_data, reduce_word, merge_pdf_files
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join as join

# Switch under the working directory
os.chdir(r'/Users/bs3667/PycharmProjects/NoiseProject')
# Define I/O directories
datadir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/TaskProgram/log/txtDat'
mydatdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/myData'
svdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults'
# set color space
OKeffee = ['#C4655A',  # Earthy red-brown
                  '#3D6C4C',  # Deep green
                  '#D09E7F',  # Soft peach
                  '#584B42',  # Warm gray
                  '#7D6E91']  # Muted lavender
# loading data
bd = load_data(datadir, 'BidTask_22*')
print(bd)
bd['Definitive'] = (bd['Group'] == bd['patch']).astype(int)
bd = bd.rename(columns={'item': 'Item'})
# Normalize the bid value by
# Rescale the bid value to the max of each individual
maxvals = bd.groupby('subID')['bid'].max()
Sublist = np.unique(bd['subID'])
bd['bidMaxScl'] = 999
for si in Sublist:
    bd.loc[bd['subID'] == si, 'bidMaxScl'] = bd[bd['subID'] == si]['bid']/maxvals[si]
# Normalize the bid value by
# Divide the value by its standard deviation
stds = bd.groupby('subID')['bid'].std()
bd['bidNorm'] = 999
for si in Sublist:
    bd.loc[bd['subID'] == si, 'bidNorm'] = bd[bd['subID'] == si]['bid']/stds[si]

# Raw scale of bid value
AnalysName = 'Individual Bid_RawScl'
bdw = bd.pivot(index=['Definitive', 'subID', 'Item'], columns='bid_times', values='bid')
bdw['BidMean'] = bdw.apply(lambda row: np.mean(row[[1, 2, 3]]), axis=1)
bdw['BidSd'] = bdw.apply(lambda row: np.std(row[[1, 2, 3]]), axis=1)
bdw['sd12'] = bdw.apply(lambda row: np.std(row[[1, 2]]), axis=1)
bdw['sd23'] = bdw.apply(lambda row: np.std(row[[2, 3]]), axis=1)
bdw['sd13'] = bdw.apply(lambda row: np.std(row[[1, 3]]), axis=1)
bdw['cv'] = bdw['BidSd'] / bdw['BidMean']
bdw = bdw.reset_index()

mt = load_data(datadir, 'MainTask_22*')
print(mt)
mt['Vaguenesscode'] -= 1
mt['Definitive'] = 1 - mt['Vaguenesscode']
# To merge the bidding variance information in to the choice matrix
IDs = {'ID1': 'V1', 'ID2': 'V2', 'ID3': 'V3'}
for ID, V in IDs.items():
    mt['Item'] = mt[ID]
    tmp = pd.merge(bdw[['subID', 'Item', 'BidSd']], mt[['subID', 'Item', 'trial']], on=['subID', 'Item'])
    tmpp = tmp.rename(columns={'BidSd': 'sd'+V})
    mt = pd.merge(mt, tmpp.drop('Item', axis=1), on=['subID', 'trial'])
mt = mt.drop('Item', axis=1)
# mt['chosenItem'] = mt['chosenItem'].fillna(99).astype(int)
mt = mt.dropna(subset=['chosenItem'])
mt['chosenItem'] = mt['chosenItem'].astype(int)
mt.to_csv(join(mydatdir, 'TrnsfrmData.csv'), index=False)
sublist = np.unique(mt.subID)
CorrectItems = pd.read_excel(r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/TaskProgram/CorrectStimuli/CorrectItems.xlsx')
CorrectItems.rename(columns={'Number': 'Item', 'Name of Item': 'name'}, inplace=True)
CorrectItems.to_csv(join(mydatdir, 'CorrectItems.csv'), index=False)

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
            indvdat = mt[mt['subID']==sublist[subj]] # & mt['TimePressure'] == "Low"
            grouped = indvdat.groupby(['V1', 'V2', 'chosenItem']).size().reset_index(name='Count')
            wide_df = grouped.pivot_table(values='Count', index=['V1', 'V2'], columns='chosenItem', fill_value=0)
            wide_df = wide_df.reset_index()
            sns.scatterplot(x="V1", y="V2", hue="chosenItem", size="Count",
                        sizes=(12, 300), alpha=.45, legend=lgd, palette=colors, data=grouped, ax=axs[r,c])
            # sns.scatterplot(x="V1", y="V2", size="Count",
            #                 sizes=(12, 300), legend=False, color="purple", data=grouped[grouped['chosenItem'] == 2], ax=axs[r, c])
            # sns.scatterplot(x="V1", y="V2", size="Count",
            #                 sizes=(12, 300), legend=False, color="red", data=grouped[grouped['chosenItem'] == 1], ax=axs[r, c])
            # sns.scatterplot(x="V1", y="V2", size="Count",
            #                 sizes=(12, 300), legend=False, color="cyan", data=grouped[grouped['chosenItem'] == 3], ax=axs[r, c])
            if lgd:
                axs[r,c].legend(fontsize='5')
            lgd=False
            axs[r,c].set_title(f'#{subj+1}, {sublist[subj]}')
            subj += 1
    plt.tight_layout()
    filename = join(svdir, 'Individual_choice', 'tgrtPairs_toSubj' + str(subj) + '.pdf')
    pdf_files.append(filename)
    plt.savefig(filename, format='pdf')
    plt.close(fig)
output_file = join(svdir, 'Individual_choice', 'tgrtPairs_Cmbnd.pdf')
merge_pdf_files(pdf_files, output_file)

pr = mt.groupby(['subID', 'chosenItem']).size().reset_index(name='Count')
prw = pr.pivot(index=['subID'], columns='chosenItem', values='Count')
prw = prw.reset_index()
prw['pr1'] = prw[1]/prw[[1,2,3]].sum(axis=1)
prw['pr2'] = prw[2]/prw[[1,2,3]].sum(axis=1)
prw['pr3'] = prw[3]/prw[[1,2,3]].sum(axis=1)
prw['subID'] = list(range(1,61))
fastbads = load_data(r'/Users/bs3667/Noise/modelfit/Results/FastBADS_Mtlb', '*Best*.txt')
fastbadsw = fastbads[fastbads['modeli']>2].pivot(index=['subID'], columns='name', values='wp')
fastbadsw  = fastbadsw.reset_index()
df = pd.merge(prw, fastbadsw, on='subID', how='inner')

g = sns.pairplot(df[['pr2', 'DN', 'dDNa', 'dDNb', 'dDNc', 'dDNd']])
for ax in g.axes.ravel()[range(1,6)]:
    ax.axvline(x=0, ls='--', linewidth=1, c='gray')
    ax.axhline(y=0.5, ls='--', linewidth=1, c='gray')
plt.tight_layout()
plt.savefig(join(svdir, 'ModelFitting', 'pr2_and_wp.pdf'), format='pdf')

# check individually
subj = 57
indvdat = mt[mt['subID']==sublist[subj]]
IDs = indvdat.groupby(['ID1', 'V1']).size().reset_index(name='Count')
grouped = indvdat.groupby(['chosenItem']).size().reset_index(name='Count')
            wide_df = grouped.pivot_table(values='Count', index=['V1', 'V2'], columns='chosenItem', fill_value=0)
sns.pairplot(indvdat[['V1','V2','V3','chosenItem']], hue='chosenItem', palette=colors)
plt.show()
plt.savefig(join(svdir, 'ModelFitting', 'IndividualCheck', f'{sublist[subj]}.pdf'), format='pdf')


# See choice consistency bwtween revealed preference and bid values
sublist = np.unique(mt.subID)
subj = 0
pdf_files = []
while subj < len(sublist):
    if subj % 4 == 0:
        fig, axs = plt.subplots(2, 2, figsize=(12, 11))
        lgd = True
    for r in range(2):
        for c in range(2):
            indvdat = mt[mt['subID']==sublist[subj]] # & mt['TimePressure'] == "Low"
            IDavail = np.unique(indvdat[['ID1','ID2','ID3']])
            trgt = np.unique(indvdat[['ID1','ID2']])
            Freqc = indvdat.groupby(['chosenID']).size().reset_index(name='Count')
            Freqc.rename(columns={'chosenID': 'Item'}, inplace=True)
            indvbdw = bdw[bdw['subID']==sublist[subj]]
            indvbdw = pd.merge(indvbdw, CorrectItems[['Item', 'name']], on='Item')
            Freqc = pd.merge(Freqc, indvbdw, on='Item')
            Freqc['Present'] = 1
            IDnonchosen = indvbdw[indvbdw["Item"].isin(IDavail) & ~indvbdw["Item"].isin(Freqc["Item"])]
            IDnonchosen["Count"] = 0
            IDnonchosen["Present"] = 0
            Freq = pd.concat([Freqc, IDnonchosen], axis=0)
            Freq['Revealed rank'] = Freq['Count'].rank(ascending=True)
            Freq['Bid rank'] = Freq['BidMean'].rank(ascending=True)
            Freq['Consistency'] = abs(Freq['Revealed rank'] - Freq['Bid rank'])<1
            Freq['trgt'] = Freq['Item'].isin(trgt)
            subset = Freq[Freq['Present']==1]
            sns.scatterplot(subset, x='BidMean', y='Count', hue='trgt', legend=lgd, ax=axs[r, c])
            item_names = subset['name'].tolist()
            bid = subset['BidMean'].tolist()
            rvl = subset['Count'].tolist()
            x_ticklabels = [f'{item}: ({val:.1f})' for item, val in zip(item_names, bid)]
            y_ticklabels = [f'{item}: ({val})' for item, val in zip(item_names, rvl)]
            x_ticks = subset['BidMean'].tolist()
            y_ticks = subset['Count'].tolist()
            axs[r, c].set_xticks(x_ticks, x_ticklabels, rotation=45, ha='right')
            axs[r, c].set_yticks(y_ticks, y_ticklabels, ha='right')
            if lgd:
                axs[r, c].legend(fontsize='5', title='is trgt')
            lgd = False
            axs[r, c].set_title(f'#{subj + 1}, {sublist[subj]}')
            subj += 1
    plt.tight_layout()
    filename = join(svdir, 'Individual_choice', 'Consistency_toSubj' + str(subj) + '.pdf')
    pdf_files.append(filename)
    plt.savefig(filename, format='pdf')
    plt.close(fig)
output_file = join(svdir, 'Individual_choice', 'ConsistencyCountBid_Cmbnd.pdf')
merge_pdf_files(pdf_files, output_file)

# plt.figure()
# sns.scatterplot(Freq, x='BidMean', y='Count')
# item_names = Freq['name'].tolist()
# x_ticks = Freq['BidMean'].tolist()
# x_ticklabels = [f'{item}: ({bid:.1f})' for item, bid in zip(item_names, x_ticks)]
# plt.xticks(x_ticks, x_ticklabels, rotation=45, ha='right')
# plt.tight_layout()
# plt.title(f'# {subj+1}, {sublist[subj]}')
# plt.show()

ax.set_xticks(x_ticks)
ax.set_xticklabels(item_names, rotation=45, ha='right')

plt.axline([0,0], slope=1, linestyle='--', color='gray')
plt.title(f'# {subj+1}, {sublist[subj]}')
plt.show()

## See preference revealed z-scored distance
sublist = np.unique(mt.subID)
subj = 0
pdf_files = []
while subj < len(sublist):
    if subj % 12 == 0:
        fig, axs = plt.subplots(4, 3, figsize=(8.5, 11))
    for r in range(4):
        for c in range(3):
            indvdat = mt[mt['subID']==sublist[subj]]
            # the revealed distance for a trinary choice is defined as the distance from the unchosen options to the chosen option
            # (directed!!!), summed in norm of -1. For example, the unchosen option A and B as value x and y and sdx and sdy, the chosen
            # option C has value z and sdz.
            # The distance from A to C is xz = (z - x)/sqrt(sdx^2 + sdz^2)
            # The distance from B to C is yz = (y - z)/sqrt(sdy^2 + sdz^2)
            # The -1 norm of the two distance values is (xz^-1 + yz^-1)^-1
            tmpV = indvdat[['V1','V2','V3']]
            tmpsd = indvdat[['sdV1', 'sdV2', 'sdV3']]
            options = np.array([1, 2, 3])
            zistance = np.zeros((len(tmpV), 1))
            for i in range(len(tmpV)):
                chosen = indvdat.iloc[i]['chosenItem']
                unchosen = options[options != chosen]
                xz = (tmpV.iloc[i][f'V{chosen}'] - tmpV.iloc[i][f'V{unchosen[0]}']) / (
                        tmpsd.iloc[i][f'sdV{chosen}'] ** 2 + tmpsd.iloc[i][f'sdV{unchosen[0]}'] ** 2) ** .5
                yz = (tmpV.iloc[i][f'V{chosen}'] - tmpV.iloc[i][f'V{unchosen[1]}']) / (
                            tmpsd.iloc[i][f'sdV{chosen}'] ** 2 + tmpsd.iloc[i][f'sdV{unchosen[1]}'] ** 2) ** .5
                zistance[i] = (xz ** -1 + yz ** -1) ** -1
                print(f'i{i}, Z {zistance[i]}')
            axs[r, c].axvline(x=0, ls='--', linewidth=1, c='gray')
            cleaned_data = zistance[np.isfinite(zistance)]
            cleaned_data = cleaned_data[(cleaned_data < 10000) & (cleaned_data > -10000)]
            # nbins = max(int(round(max(cleaned_data) - min(cleaned_data))), 1)
            # bars = axs[r,c].hist(cleaned_data, bins=nbins, density=True, color='skyblue', edgecolor='None', alpha=0.7)
            # sns.kdeplot(cleaned_data, color='red', linewidth=2, ax=axs[r, c])
            sns.histplot(cleaned_data, kde=True, color=OKeffee[0], edgecolor='None', legend=False, ax=axs[r, c])
            lines = axs[r, c].get_lines()
            if len(lines) > 1:
                line = lines[1]
                x_data, y_data = line.get_data()
                x_mask = x_data[(y_data > max(y_data)*.01)]
                axs[r, c].set_xlim(min(x_mask), max(x_mask))
                axs[r, c].set_ylim(0, max(y_data)*1.1)
            axs[r, c].set_xlabel('Zistance')
            axs[r, c].set_ylabel('Frequency / Density')
            axs[r, c].set_title(f'#{subj + 1}, {sublist[subj]}')
            subj += 1
    plt.tight_layout()
    filename = join(svdir, 'Individual_choice', 'Zistance_toSubj' + str(subj) + '.pdf')
    pdf_files.append(filename)
    plt.savefig(filename, format='pdf')
    plt.close(fig)
output_file = join(svdir, 'Individual_choice', 'Zistance_Cmbnd.pdf')
merge_pdf_files(pdf_files, output_file)

# See RT skewness
sublist = np.unique(mt.subID)
dtype = [('subID', int), ('FisherPearson', float), ('TimePressure', 'U30')]  # Adjust data types as needed
Skewness = np.empty(0, dtype=dtype)
subj = 0
pdf_files = []
while subj < len(sublist):
    if subj % 12 == 0:
        fig, axs = plt.subplots(4, 3, figsize=(8.5, 11))
        lgd = True
    for r in range(4):
        for c in range(3):
            RTLow = mt[(mt['subID']==sublist[subj]) & (mt['TimePressure'] == 'Low')]['RT']
            RTHigh = mt[(mt['subID']==sublist[subj]) & (mt['TimePressure'] == 'High')]['RT']
            data_to_append_low = (sublist[subj], ss.skew(RTLow, axis=None, bias=True, nan_policy='omit'), 'Low')
            Skewness = np.append(Skewness, np.array(data_to_append_low, dtype=dtype))
            data_to_append_high = (sublist[subj], ss.skew(RTHigh, axis=None, bias=True, nan_policy='omit'), 'High')
            Skewness = np.append(Skewness, np.array(data_to_append_high, dtype=dtype))
            num_bins = 50
            bins = np.linspace(min([RTLow.min(), RTHigh.min()]), max([RTLow.max(), RTHigh.max()]), num_bins)
            sns.set_palette('Paired')
            sns.histplot(RTHigh, bins=bins, kde=True, alpha=0.5, label='1.5s', ax=axs[r, c])
            sns.histplot(RTLow, bins=bins, kde=True, alpha=0.5, label='10s', ax=axs[r, c])
            if lgd:
                axs[r,c].legend(fontsize='5')
            lgd=False
            axs[r,c].set_title(f'#{subj+1}, {sublist[subj]}')
            subj += 1
    plt.tight_layout()
    filename = join(svdir, 'Individual_choice', 'IndvRTSkewness_toSubj' + str(subj) + '.pdf')
    pdf_files.append(filename)
    plt.savefig(filename, format='pdf')
    plt.close(fig)
output_file = join(svdir, 'Individual_choice', 'IndvRTSkewness_Cmbnd.pdf')
merge_pdf_files(pdf_files, output_file)
df = pd.DataFrame(Skewness)

df.to_csv(join(svdir, 'Individual_choice', 'IndvRTSkewness.csv'), index=False)
df = pd.read_csv(join(svdir, 'Individual_choice', 'IndvRTSkewness.csv'))
fig = plt.figure(figsize=(3, 3))
sns.set_palette('Paired')
plt.plot([min(df['FisherPearson']), max(df['FisherPearson'])],
         [min(df['FisherPearson']), max(df['FisherPearson'])],
         color='gray', linestyle='--', label='Diagonal Line', zorder=1)
plt.scatter(df[df['TimePressure'] == 'Low']['FisherPearson'],
            df[df['TimePressure'] == 'High']['FisherPearson'], alpha=.8, s=7)
plt.title('RT skewness (Fisher-Pearson)')
plt.xlabel('Low time pressure')
plt.ylabel('High time pressure')
plt.tight_layout()
filename = join(svdir, 'Individual_choice', 'IndvRTSkewness.pdf')
plt.savefig(filename, format='pdf')
plt.close(fig)
