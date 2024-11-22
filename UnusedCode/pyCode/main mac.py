import os
from myFunctions import load_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join as join

# Switch under the working directory
os.chdir(r'/Users/bs3667/PycharmProjects/NoiseProject')
# Define I/O directories

datadir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/TaskProgram/log/txtDat'
svdir = r'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults'

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
# Max scaled of bid value
AnalysName = 'Individual Bid_MaxScl'
bdw = bd.pivot(index=['Definitive', 'subID', 'Item'], columns='bid_times', values='bidMaxScl')
#  Divide standard deviation of bid value
AnalysName = 'Individual Bid_Norm'
bdw = bd.pivot(index=['Definitive', 'subID', 'Item'], columns='bid_times', values='bidNorm')

bdw['BidMean'] = bdw.apply(lambda row: np.mean(row[[1, 2, 3]]), axis=1)
bdw['BidSd'] = bdw.apply(lambda row: np.std(row[[1, 2, 3]]), axis=1)
bdw['sd12'] = bdw.apply(lambda row: np.std(row[[1, 2]]), axis=1)
bdw['sd23'] = bdw.apply(lambda row: np.std(row[[2, 3]]), axis=1)
bdw['sd13'] = bdw.apply(lambda row: np.std(row[[1, 3]]), axis=1)
bdw = bdw.reset_index()
bdw['fanof'] = bdw['BidSd'] / bdw['BidMean']

sns.set_palette('pastel')
sns.lmplot(x='BidMean', y='BidSd', hue='Definitive', data=bdw, scatter_kws={'alpha': 0.4})
plt.show()
plt.savefig(join(svdir, AnalysName, 'Bid Variance.pdf'), format='pdf')
model = smf.mixedlm("BidSd ~ BidMean*Definitive", data=bdw, groups=bdw["subID"], re_formula="~ 1 + BidMean")
result = model.fit()
print(result.summary())

# Visualize the bidding behavior over three times
sns.set_palette('rocket', 3)
subj = 0
Sublist = np.unique(bdw['subID'])
bdw['Rank'] = 99
while subj < len(Sublist):
    if subj % 12 == 0:
        fig, axs = plt.subplots(4, 3, figsize=(8.5, 11))
    for r in range(4):
        for c in range(3):
            df = bdw[bdw['subID'] == Sublist[subj]][[1, 2, 3, 'BidMean']].sort_values('BidMean')
            df['Rank'] = range(1, len(df) + 1)
            bdw.loc[(bdw['subID'] == Sublist[subj]), "Rank"] = df["Rank"]
            # Plot variable 1, 2, 3 as a function of the rank
            axs[r, c].scatter(df['Rank'], df[1], s=10, label='Bid 1')
            axs[r, c].scatter(df['Rank'], df[2], s=10, label='Bid 2')
            axs[r, c].scatter(df['Rank'], df[3], s=10, label='Bid 3')
            if r==3:
                axs[r, c].set_xlabel('Rank of BidMean')
            if c==0:
                axs[r, c].set_ylabel('Bid')
            axs[r, c].set_title(Sublist[subj])
            if (r==0) & (c==0):
                axs[r, c].legend()
            plt.show()
            subj += 1
    fig.tight_layout()
    plt.savefig(join(svdir, AnalysName, 'Bid 3 times_toSubj' +str(subj)+'.pdf'), format='pdf')

# To visualize the group pattern
mean_df = bdw.groupby(['Rank', 'subID']).mean().reset_index()
long_format_df = pd.melt(mean_df, id_vars=['Rank', 'subID'], value_vars=[1, 2, 3])
plt.figure(figsize=(10, 6))
sns.barplot(data=long_format_df, x='Rank', y='value', hue='bid_times', errorbar=None)
plt.xlabel('Items\' bid rank')
plt.ylabel('Bid value')
plt.title('Bid 1, 2, and 3 averaged over subjects')
plt.show()
plt.savefig(join(svdir, AnalysName, 'Bid 3 times_ItemAvg.pdf'), format='pdf')
mean_df = bdw.groupby(['Rank', 'subID']).mean().reset_index()
long_format_df = pd.melt(mean_df, id_vars=['Rank', 'subID'], value_vars=[1, 2, 3])
plt.figure(figsize=(11, 6))
sns.barplot(data=long_format_df, x='subID', y='value', hue='bid_times', errorbar=None)
plt.xticks(rotation=90)
plt.xlabel('Subject ID')
plt.ylabel('Bid value')
plt.title('Bid 1, 2, and 3 averaged over subjects')
plt.tight_layout()
plt.show()
plt.savefig(join(svdir, AnalysName, 'Bid 3 times_IndvAvg.pdf'), format='pdf')
# do statistics
from statsmodels.formula.api import ols
# Reshape the data to long format if necessary
df_long = pd.melt(bdw, id_vars=['subID', 'Rank'],
                  value_vars=[1, 2, 3],
                  var_name='bid_times',
                  value_name='bid')
# Perform two-way ANOVA
model = ols('bid ~ C(bid_times) + C(subID) + C(Rank) + C(bid_times):C(subID) + C(bid_times):C(Rank)', data=df_long).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Inspired by the statistics, which shows interaction between bid_times and subID
bdw1 = bdw[~(bdw[[1, 2, 3]] == 0).all(axis=1)]
bdw1.loc[:, [1, 2, 3]] = bdw1[[1, 2, 3]].apply(lambda x: x / x.max(), axis=1)
# Visualize the bidding behavior over three times
sns.set_palette('rocket', 3)
subj = 0
Sublist = np.unique(bdw1['subID'])
while subj < len(Sublist):
    if subj % 12 == 0:
        fig, axs = plt.subplots(4, 3, figsize=(8.5, 11))
    for r in range(4):
        for c in range(3):
            df = bdw1[bdw1['subID'] == Sublist[subj]][[1, 2, 3, 'BidMean', 'Rank']].sort_values('BidMean')
            # Plot variable 1, 2, 3 as a function of the rank
            axs[r, c].scatter(df['Rank'], df[1], s=10, label='Bid 1')
            axs[r, c].scatter(df['Rank'], df[2], s=10, label='Bid 2')
            axs[r, c].scatter(df['Rank'], df[3], s=10, label='Bid 3')
            if r==3:
                axs[r, c].set_xlabel('Rank of BidMean')
            if c==0:
                axs[r, c].set_ylabel('Bid value rescaled to the max of three bids')
            axs[r, c].set_title(Sublist[subj])
            if (r==0) & (c==0):
                axs[r, c].legend()
            plt.show()
            subj += 1
    fig.tight_layout()
    plt.savefig(join(svdir, AnalysName, 'Bid 3 times_MaxScl_toSubj' +str(subj)+'.pdf'), format='pdf')

mean_df = bdw1.groupby(['Rank', 'subID']).mean().reset_index()
long_format_df = pd.melt(mean_df, id_vars=['Rank', 'subID'], value_vars=[1, 2, 3])
plt.figure(figsize=(10, 6))
sns.barplot(data=long_format_df, x='Rank', y='value', hue='bid_times', errorbar=None)
plt.xlabel('Items\' bid rank')
plt.ylabel('Bid value rescaled to the max of three bids')
plt.title('Bid 1, 2, and 3 averaged over subjects')
plt.show()
plt.savefig(join(svdir, AnalysName, 'Bid 3 times_MaxScl_ItemAvg.pdf'), format='pdf')
mean_df = bdw1.groupby(['Rank', 'subID']).mean().reset_index()
long_format_df = pd.melt(mean_df, id_vars=['Rank', 'subID'], value_vars=[1, 2, 3])
plt.figure(figsize=(11, 6))
sns.barplot(data=long_format_df, x='subID', y='value', hue='bid_times', errorbar=None)
plt.xticks(rotation=90)
plt.xlabel('Subject ID')
plt.ylabel('Bid value rescaled to the max of three bids')
plt.title('Bid 1, 2, and 3 averaged over subjects')
plt.tight_layout()
plt.show()
plt.savefig(join(svdir, AnalysName, 'Bid 3 times_MaxScl_IndvAvg.pdf'), format='pdf')
df_long = pd.melt(bdw1, id_vars=['subID', 'Rank'],
                  value_vars=[1, 2, 3],
                  var_name='bid_times',
                  value_name='bid')
# Perform two-way ANOVA
model = ols('bid ~ C(bid_times) + C(subID) + C(Rank) + C(bid_times):C(subID) + C(bid_times):C(Rank)', data=df_long).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Clustering the patterns of the subjects
from sklearn.cluster import KMeans
# Create a new DataFrame with the average values of columns 1, 2, 3 for each subject
df_avg = bdw1.groupby('subID')[[1, 2, 3]].mean()
# Show individual pattern
sns.clustermap(df_avg)
plt.savefig(join(svdir, AnalysName, 'Bid 3 times_MaxScl_Clust.pdf'), format='pdf')

df_avg.reset_index()
# Specify the number of clusters (change it according to your needs)
n_clusters = 3
# Create a KMeans instance with n_clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
# Fit the model to your data and predict the cluster labels
df_avg['Cluster'] = kmeans.fit_predict(df_avg[[1, 2, 3]])
df_avg = df_avg.reset_index()
# Print out the result
print(df_avg)
# If you want to visualize the result
sns.set_palette('pastel')
plt.figure()
for i in range(n_clusters):
    cluster = df_avg[df_avg['Cluster'] == i]
    plt.plot(cluster[[1, 2, 3]].mean(), 'o-',
             linewidth=len(cluster['subID'])**.5, label=f'Cluster {i+1} (N = {len(cluster)})')
plt.xlabel('Bid times')
plt.ylabel('Bid value rescaled to the max of three bids')
plt.legend()
plt.show()
plt.savefig(join(svdir, AnalysName, 'Bid 3 times_MaxScl_ClustPattern.pdf'), format='pdf')

# Check the bidding variance
# Visualize the bidding behavior over three times
sns.set_palette('rocket', 3)
subj = 0
Sublist = np.unique(bdw['subID'])
while subj < len(Sublist):
    if subj % 12 == 0:
        fig, axs = plt.subplots(4, 3, figsize=(8.5, 11))
    for r in range(4):
        for c in range(3):
            df = bdw[bdw['subID'] == Sublist[subj]][['sd23', 'sd12', 'sd13', 'BidMean', 'Rank']].sort_values('BidMean')
            # Plot variable 1, 2, 3 as a function of the rank
            axs[r, c].scatter(df['Rank'], df['sd23'], s=10, label='Bid variance 2 and 3')
            axs[r, c].scatter(df['Rank'], df['sd12'], s=10, label='Bid variance 1 and 2')
            axs[r, c].scatter(df['Rank'], df['sd13'], s=10, label='Bid variance 1 and 3')
            if r==3:
                axs[r, c].set_xlabel('Rank of BidMean')
            if c==0:
                axs[r, c].set_ylabel('Bid variance')
            axs[r, c].set_title(Sublist[subj])
            if (r==0) & (c==0):
                axs[r, c].legend()
            plt.show()
            subj += 1
    fig.tight_layout()
    plt.savefig(join(svdir, AnalysName, 'Bid variance_toSubj' +str(subj)+'.pdf'), format='pdf')

mean_df = bdw.groupby(['Rank', 'subID']).mean().reset_index()
long_format_df = pd.melt(mean_df, id_vars=['Rank', 'subID'], value_vars=['sd23', 'sd12', 'sd13'])
plt.figure(figsize=(10, 6))
sns.barplot(data=long_format_df, x='Rank', y='value', hue='bid_times', errorbar=None)
plt.xlabel('Items\' bid rank')
plt.ylabel('Bid variance')
plt.title('Bid variance averaged over subjects')
plt.show()
plt.savefig(join(svdir, AnalysName, 'Bid variance_ItemAvg.pdf'), format='pdf')
mean_df = bdw.groupby(['Rank', 'subID']).mean().reset_index()
long_format_df = pd.melt(mean_df, id_vars=['Rank', 'subID'], value_vars=['sd23', 'sd12', 'sd13'])
plt.figure(figsize=(11, 6))
sns.barplot(data=long_format_df, x='subID', y='value', hue='bid_times', errorbar=None)
plt.xticks(rotation=90)
plt.xlabel('Subject ID')
plt.ylabel('Bid variance')
plt.title('Bid variance averaged over subjects')
plt.tight_layout()
plt.show()
plt.savefig(join(svdir, AnalysName, 'Bid variance_IndvAvg.pdf'), format='pdf')
df_long = pd.melt(bdw1, id_vars=['subID', 'Rank'],
                  value_vars=['sd23', 'sd12', 'sd13'],
                  var_name='bid_times',
                  value_name='bid')
# Perform two-way ANOVA
model = ols('bid ~ C(bid_times) + C(subID) + C(Rank) + C(bid_times):C(subID) + C(bid_times):C(Rank)', data=df_long).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Clustering the patterns of the subjects
from sklearn.cluster import KMeans
# Create a new DataFrame with the average values of columns 1, 2, 3 for each subject
df_avg = bdw1.groupby('subID')[['sd23', 'sd12', 'sd13']].mean()
# Show individual pattern
sns.clustermap(df_avg)
plt.savefig(join(svdir, AnalysName, 'Bid variance_Clust.pdf'), format='pdf')

df_avg = df_avg.reset_index()
# Specify the number of clusters (change it according to your needs)
n_clusters = 3
# Create a KMeans instance with n_clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
# Fit the model to your data and predict the cluster labels
df_avg['Cluster'] = kmeans.fit_predict(df_avg[['sd23', 'sd12', 'sd13']])
df_avg = df_avg.reset_index()
# Print out the result
print(df_avg)
# If you want to visualize the result
sns.set_palette('pastel')
plt.figure()
for i in range(n_clusters):
    cluster = df_avg[df_avg['Cluster'] == i]
    plt.plot(cluster[['sd23', 'sd12', 'sd13']].mean(), 'o-',
             linewidth=len(cluster['subID'])**.5, label=f'Cluster {i+1} (N = {len(cluster)})')
plt.xlabel('Bid times')
plt.ylabel('Bid value rescaled to the max of three bids')
plt.legend()
plt.show()
plt.savefig(join(svdir, AnalysName, 'Bid 3 times_MaxScl_ClustPattern.pdf'), format='pdf')


# To load choice data
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

Singulardata = mt[(mt['sdV1']==0) & (mt['sdV2']==0)]
Agr = Singulardata.groupby(['subID', 'V1', 'V2'])['trial'].count().reset_index()
sns.relplot(x='V1', y='V2', hue='subID', size='trial', sizes=(80, 800), alpha=.5, palette='muted', height=6, data=Agr)
plt.savefig(join(svdir, 'SnglrData_Ntrials.pdf'), format='pdf')

Agr = Singulardata.groupby(['subID', 'chosenItem'])['trial'].count().reset_index().pivot(index='subID', columns='chosenItem', values='trial').fillna(0)
Agr.plot(kind='bar', stacked=True, figsize=(10, 7), xlabel='SubID', ylabel='Trial count', title='Trial counts by chosen item')
plt.legend(title='Chosen Item')
plt.tight_layout()
plt.show()
plt.savefig(join(svdir, 'SnglrData_Trialcounts.pdf'), format='pdf')


# Maximum likelihood fitting to the choice behavior by using
# Q1: The raw scale bidding values vs. the rescaled bidding values, should be the same under representation noise. So try raw scale first
# Q2: The selection noise is additive or multiplicative? Try additive first
from scipy.stats import norm
from scipy.optimize import minimize
AnalysName = join('ModelFitting')
os.mkdir(join(svdir, AnalysName))

# Model 1. without representation noise, but only selection noise
def neg_ll_indv1(x, dat):
    # let's assume the value of each item has no noise
    # but there is a fixed noise during the selection process
    # so that we can calculate the probability of choosing one from the other two options
    # by computing the probability of the largest value of the random drawn is from which option
    M_diff, SD_diff = dat['V1'] - dat['V2'], 2**.5*x
    # calculate the probability
    prob12 = norm(M_diff, SD_diff).sf(0)  # sf(x) is 1 - cdf(x)
    prob21 = 1 - prob12
    M_diff, SD_diff = dat['V1'] - dat['V3'], 2**.5*x
    prob13 = norm(M_diff, SD_diff).sf(0)  # sf(x) is 1 - cdf(x)
    prob31 = 1 - prob13
    M_diff, SD_diff = dat['V2'] - dat['V3'], 2**.5*x
    prob23 = norm(M_diff, SD_diff).sf(0)  # sf(x) is 1 - cdf(x)
    prob32 = 1 - prob23
    p1=prob12*prob13
    p2=prob21*prob23
    p3=prob31*prob32
    df=pd.DataFrame({1: p1, 2: p2, 3: p3})
    col_labels = dat['chosenItem'].astype(int)
    col_indices = [df.columns.get_loc(c) for c in col_labels]
    ll = sum(np.log(max(df.iloc[i, col], np.finfo(float).eps)) for i, col in enumerate(col_indices))
    return -ll

# Model 2. with representation noise and selection noise, but no normalization
def neg_ll_indv2(x, dat):
    # let's assume the bid value has noise as measured in bid variance
    # there is still a fixed noise during the selection process
    # so that we can calculate the probability of choosing one from the other two options
    # by computing the probability of the largest value of the random drawn is from which option
    M_diff, SD_diff = dat['V1'] - dat['V2'], (2*x**2+dat['sdV1']**2+dat['sdV2']**2)**0.5
    # calculate the probability
    prob12 = norm(M_diff, SD_diff).sf(0)  # sf(x) is 1 - cdf(x)
    prob21 = 1 - prob12
    M_diff, SD_diff = dat['V1'] - dat['V3'], (2*x**2+dat['sdV1']**2+dat['sdV3']**2)**0.5
    prob13 = norm(M_diff, SD_diff).sf(0)  # sf(x) is 1 - cdf(x)
    prob31 = 1 - prob13
    M_diff, SD_diff = dat['V2'] - dat['V3'], (2*x**2+dat['sdV2']**2+dat['sdV3']**2)**0.5
    prob23 = norm(M_diff, SD_diff).sf(0)  # sf(x) is 1 - cdf(x)
    prob32 = 1 - prob23
    p1=prob12*prob13
    p2=prob21*prob23
    p3=prob31*prob32
    df=pd.DataFrame({1: p1, 2: p2, 3: p3})
    col_labels = dat['chosenItem'].astype(int)
    col_indices = [df.columns.get_loc(c) for c in col_labels]
    ll = sum(np.log(max(df.iloc[i, col], np.finfo(float).eps)) for i, col in enumerate(col_indices))
    return -ll

# Model 3. with representation noise under divisive normalization, and selection noise
def neg_ll_indv3(x, dat):
    # let's assume the bid value with bid variance is divisively normalized
    # there is still a fixed noise during the selection process
    # so that we can calculate the probability of choosing one from the other two options
    # by computing the probability of the largest value of the random drawn is from which option
    M_diff, SD_diff = dat['V1'] - dat['V2'], (2*x**2+dat['sdV1']**2+dat['sdV2']**2)**0.5
    # calculate the probability
    prob12 = norm(M_diff, SD_diff).sf(0)  # sf(x) is 1 - cdf(x)
    prob21 = 1 - prob12
    M_diff, SD_diff = dat['V1'] - dat['V3'], (2*x**2+dat['sdV1']**2+dat['sdV3']**2)**0.5
    prob13 = norm(M_diff, SD_diff).sf(0)  # sf(x) is 1 - cdf(x)
    prob31 = 1 - prob13
    M_diff, SD_diff = dat['V2'] - dat['V3'], (2*x**2+dat['sdV2']**2+dat['sdV3']**2)**0.5
    prob23 = norm(M_diff, SD_diff).sf(0)  # sf(x) is 1 - cdf(x)
    prob32 = 1 - prob23
    p1=prob12*prob13
    p2=prob21*prob23
    p3=prob31*prob32
    df=pd.DataFrame({1: p1, 2: p2, 3: p3})
    col_labels = dat['chosenItem'].astype(int)
    col_indices = [df.columns.get_loc(c) for c in col_labels]
    ll = sum(np.log(max(df.iloc[i, col], np.finfo(float).eps)) for i, col in enumerate(col_indices))
    return -ll

# Fit and plot
sns.set_palette('Paired')
sublist = np.unique(mt['subID'])
Rslts = pd.DataFrame(columns=['subID', 'Model', 'sigma', 'nll'])
subj = 0
while subj < len(sublist):
    if subj % 12 == 0:
        fig, axs = plt.subplots(4, 3, figsize=(8.5, 11))
    for r in range(4):
        for c in range(3):
            dat = mt[mt['subID'] == sublist[subj]]
            # Initial guess for x
            x0 = 1
            # Call the optimizer
            res1 = minimize(neg_ll_indv1, x0, args=dat, method='BFGS')
            new_row = pd.DataFrame({'subID': sublist[subj], 'Model': 1, 'sigma': [res1.x], 'nll': [res1.fun]})
            Rslts = pd.concat([Rslts, new_row], ignore_index=True)

            res2 = minimize(neg_ll_indv2, x0, args=dat, method='BFGS')
            new_row = pd.DataFrame({'subID': sublist[subj], 'Model': 2, 'sigma': [res2.x], 'nll': [res2.fun]})
            Rslts = pd.concat([Rslts, new_row], ignore_index=True)

            axs[r, c].set_title(sublist[subj])
            if not np.isnan(res1.fun):
                xvec = np.arange(.1, 2 * res1.x, res1.x/200)
                f = [neg_ll_indv1(x, dat) for x in xvec]
                axs[r, c].plot(xvec, f, 'r', label='Model 1')
                axs[r, c].plot(res1.x, res1.fun, 'r.')
            if not np.isnan(res2.fun):
                xvec = np.arange(.1, 2 * res2.x, res2.x/200)
                f = [neg_ll_indv2(x, dat) for x in xvec]
                axs[r, c].plot(xvec, f, 'b', label='Model 2')
                axs[r, c].plot(res2.x, res2.fun, 'b.')
            if r == 3:
                axs[r, c].set_xlabel(r'$\sigma$')
            if c == 0:
                axs[r, c].set_ylabel('Negative loglikelihood')
            if (r == 0) & (c == 0):
                axs[r, c].legend()
            subj += 1
    plt.show()
    fig.tight_layout()
    plt.savefig(join(svdir, AnalysName, 'Optimization_toSubj' + str(subj) + '.pdf'), format='pdf')
Rslts.to_csv(join(svdir, AnalysName,'Rslts.txt'), sep='\t', index=False)




# Model comparison
df = pd.merge(fitvals1[['subID', 'sigma', 'nll']], fitvals2[['subID', 'sigma', 'nll']], on=['subID'])
df['dnll'] = df['nll_y'] - df['nll_x']
long_format_df = pd.melt(df, id_vars=['subID'], value_vars=['nll_x', 'nll_y'])
sns.barplot(data=long_format_df, x='subID', y='value', hue='variable')
plt.xticks(rotation=90)
df['dnll'].dropna().mean() # nll, smaller is better
# Negative means model 2 better
# Positive means model 1 better.
# This mean value is negative, 22.8976581247502553
plt.figure()
sns.barplot(data=df.dropna(subset='dnll'), x='subID', y='dnll')
plt.xticks(rotation=90)