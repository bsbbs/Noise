import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as join
import statsmodels.api as sm
import statsmodels.formula.api as smf
from myFunctions import load_data, McFadden, Mdl2, DN, dDN
from noisyopt import minimizeCompass

# Switch under the working directory
os.chdir(r'C:\Users\Bo\PycharmProjects\NoiseProject')
# Define I/O directories
datadir = r'C:\Users\Bo\Dropbox (NYU Langone Health)\CESS-Bo\TaskProgram\log\txtDat'
svdir = r'C:\Users\Bo\Dropbox (NYU Langone Health)\CESS-Bo\pyResults'

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
bdw = bd.pivot(index=['Definitive', 'subID', 'Item'], columns='bid_times', values='bid')
bdw['BidMean'] = bdw.apply(lambda row: np.mean(row[[1, 2, 3]]), axis=1)
bdw['BidSd'] = bdw.std(axis=1)
bdw['sd12'] = bdw.apply(lambda row: np.std(row[[1, 2]]), axis=1)
bdw['sd23'] = bdw.apply(lambda row: np.std(row[[2, 3]]), axis=1)
bdw['sd13'] = bdw.apply(lambda row: np.std(row[[1, 3]]), axis=1)
bdw['fanof'] = bdw['BidSd'] / bdw['BidMean']
bdw = bdw.reset_index()
print(bdw)

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
mt = mt.dropna(subset=['chosenItem'])
mt['chosenItem'] = mt['chosenItem'].astype(int)

# Maximum likelihood fitting to the choice behavior by using
# Q1: The raw scale bidding values vs. the rescaled bidding values, should be the same under representation noise. So try raw scale first
# Q2: The selection noise is additive or multiplicative? Try additive first

# Model 1. without representation noise, but only selection noise
def neg_ll_indv1(x, dat):
    # let's assume the value of each item has no noise
    # but there is a fixed noise during the selection process
    # so that we can calculate the probability of choosing one from the other two options
    # by computing the probability of the largest value of the random drawn is from which option
    data = dat[['V1', 'V2', 'V3']]
    eta = x[0]
    probs = [McFadden(row.V1, row.V2, row.V3, eta) for row in
             data.itertuples()]
    probs_stacked = torch.stack(probs).cpu()
    ll = sum(np.log(max(probs_stacked[i][j - 1], np.finfo(float).eps)) for i, j in enumerate(dat['chosenItem']))
    return -ll.item()

# Model 2. with representation noise and selection noise, but no normalization
def neg_ll_indv2(x, dat):
    # let's assume the bid value has noise as measured in bid variance
    # there is still a fixed noise during the selection process
    # so that we can calculate the probability of choosing one from the other two options
    # by computing the probability of the largest value of the random drawn is from which option
    data = dat[['V1', 'V2', 'V3', 'sdV1', 'sdV2', 'sdV3']]
    eta = x[0]
    probs = [Mdl2(row.V1, row.V2, row.V3, row.sdV1, row.sdV2, row.sdV3, eta) for row in
             data.itertuples()]
    probs_stacked = torch.stack(probs).cpu()
    ll = sum(np.log(max(probs_stacked[i][j - 1], np.finfo(float).eps)) for i, j in enumerate(dat['chosenItem']))
    return -ll.item()


# Model 3. with no noise under divisive normalization, and adding with selection noise
def neg_ll_indv3(y, dat):
    # let's assume the mean value of the bid is divisively normalized
    # there is still a fixed noise during the selection process
    # so that we can calculate the probability of choosing one from the other two options
    # by computing the probability of the largest value of the random drawn is from which option
    # y[0] = Mp, y[1] = wp
    data = dat[['V1', 'V2', 'V3']]
    probs = [DN(row.V1, row.V2, row.V3, y[0], y[1]) for row in
             data.itertuples()]
    probs_stacked = torch.stack(probs).cpu()
    ll = sum(np.log(max(probs_stacked[i][j - 1], np.finfo(float).eps)) for i, j in enumerate(dat['chosenItem']))
    return -ll.item()

# Model 4. with representation noise under divisive normalization, and selection noise
def neg_ll_indv4(y, dat):
    # let's assume the bid value with bid variance is divisively normalized
    # there is still a fixed noise during the selection process
    # so that we can calculate the probability of choosing one from the other two options
    # by computing the probability of the largest value of the random drawn is from which option
    # y[0] = Mp, y[1] = wp
    mode = 'cutoff'
    data = dat[['V1', 'V2', 'V3', 'sdV1', 'sdV2', 'sdV3']]
    eta = 1
    probs = [dDN(row.V1, row.V2, row.V3, row.sdV1, row.sdV2, row.sdV3, y[0], y[1], mode) for row in data.itertuples()]
    probs_stacked = torch.stack(probs).cpu()
    ll = sum(np.log(max(probs_stacked[i][j-1], np.finfo(float).eps)) for i, j in enumerate(dat['chosenItem']))
    return -ll.item()
def neg_ll_indv4b(y, dat):
    # let's assume the bid value with bid variance is divisively normalized
    # there is still a fixed noise during the selection process
    # so that we can calculate the probability of choosing one from the other two options
    # by computing the probability of the largest value of the random drawn is from which option
    # y[0] = Mp, y[1] = wp
    mode = 'cutoff'
    data = dat[['V1', 'V2', 'V3', 'sdV1', 'sdV2', 'sdV3']]
    eta = 1
    probs = [dDNb(row.V1, row.V2, row.V3, row.sdV1, row.sdV2, row.sdV3, y[0], y[1], mode) for row in data.itertuples()]
    probs_stacked = torch.stack(probs).cpu()
    ll = sum(np.log(max(probs_stacked[i][j-1], np.finfo(float).eps)) for i, j in enumerate(dat['chosenItem']))
    return -ll.item()

# Start fitting
from scipy.stats import norm
from scipy.optimize import minimize
from pyvbmc import VBMC
from pybads import BADS
import torch
AnalysName = join('ModelFitting')
if not os.path.exists(join(svdir, AnalysName)):
    os.mkdir(join(svdir, AnalysName))
options = {
    "uncertainty_handling": True,
    "max_fun_evals": 3000,
    "noise_final_samples": 30
}
sublist = np.unique(mt['subID'])
# Rslts = pd.DataFrame(columns=['subID', 'Model', 'sigma', 'M', 'w', 'nll'])
Rslts = pd.DataFrame(columns=['subID', 'Model', 'eta', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'func_count'])
subj = 0
while subj < len(sublist):
    if subj % 12 == 0:
        fig, axs = plt.subplots(4, 3, figsize=(8.5, 11))
    for r in range(4):
        for c in range(3):
            print(f'Subject {subj+1}:', end='\t')
            dat = mt[mt['subID'] == sublist[subj]]
            # # Model 1
            # LB = 0
            # UB = 1000
            # PLB = 1.4
            # PUB = 100
            # x0 = np.array(np.random.uniform(PLB, PUB))
            # res1 = minimize(neg_ll_indv1, x0[0], args=dat, bounds=bounds, method='Nelder-Mead', tol = 0.1, options={'maxiter': 1000})
            # vbmc1 = VBMC(neg_ll_indv1, x0, LB, UB, PLB, PUB)
            # vbmc1 = VBMC(neg_ll_indv1, 3.5, LB, UB, PLB, PUB)
            # vp, results = vbmc1.optimize()
            # output = join(svdir, AnalysName, 'Model1', f'Nelder-Mead_{sublist[subj]}.npy')
            # np.save(output, res1)
            # new_row = pd.DataFrame({'subID': sublist[subj], 'Model': 1, 'eta': [res1.x], 'Mp': float('nan'), 'wp': float('nan'),
            #                         'nll': [res1.fun], 'success': res1.success, 'nit': res1.nit})
            # Rslts = pd.concat([Rslts, new_row], ignore_index=True)
            # print(f'Model 1 nll={res1.fun}', end='\t')
            #
            # # Model 2
            # def noisy_object(x):
            #     nll = neg_ll_indv2(x, dat)
            #     return nll
            # LB = np.array([0])
            # UB = np.array([1000])
            # PLB = np.array([1.4])
            # PUB = np.array([100])
            # x0 = np.array([np.random.uniform(PLB[i], PUB[i]) for i in range(1)])
            # bads = BADS(noisy_object, x0, LB, UB, PLB, PUB, options=options)
            # res2 = bads.optimize()
            # output = join(svdir, AnalysName, 'Model2', f'PyBADS_{sublist[subj]}.npy')
            # np.save(output, res2)
            # new_row = pd.DataFrame(
            #     {'subID': sublist[subj], 'Model': 2, 'eta': res2['x'][0], 'Mp': float('nan'), 'wp': float('nan'),
            #      'nll': res2['fval'], 'nllsd': res2['fsd'], 'success': res2['success'],
            #      'func_count': res2['func_count']})
            # Rslts = pd.concat([Rslts, new_row], ignore_index=True)
            # print(f"Model 2 nll={res2['fval']}")

            # Model 3
            def noisy_object(x):
                nll = neg_ll_indv3(x, dat)
                return nll
            LB = np.array([0, -1])
            UB = np.array([1000, 1])
            PLB = np.array([1.4, 0.1])
            PUB = np.array([100, .5])
            x0 = np.array([np.random.uniform(PLB[i], PUB[i]) for i in range(2)])
            bads = BADS(noisy_object, x0, LB, UB, PLB, PUB, options=options)
            res3 = bads.optimize()
            output = join(svdir, AnalysName, 'Model3', f'PyBADS_{sublist[subj]}.npy')
            np.save(output, res3)
            new_row = pd.DataFrame(
                {'subID': sublist[subj], 'Model': 3, 'eta': [1], 'Mp': res3['x'][0], 'wp': res3['x'][1],
                 'nll': res3['fval'], 'nllsd': res3['fsd'], 'success': res3['success'],
                 'func_count': res3['func_count']})
            Rslts = pd.concat([Rslts, new_row], ignore_index=True)
            print(f"Model 3 nll={res3['fval']}", end='\t')

            # Model 4
            def noisy_object(x):
                nll = neg_ll_indv4(x, dat)
                return nll
            LB = np.array([0, -1])
            UB = np.array([1000, 1])
            PLB = np.array([1.4, 0.1])
            PUB = np.array([100, .5])
            x0 = np.array([np.random.uniform(PLB[i], PUB[i]) for i in range(2)])
            bads = BADS(noisy_object, x0, LB, UB, PLB, PUB, options=options)
            res4 = bads.optimize()
            output = join(svdir, AnalysName, 'Model4', f'PyBADS_{sublist[subj]}.npy')
            np.save(output, res4)
            new_row = pd.DataFrame(
                {'subID': sublist[subj], 'Model': 4, 'eta': [1], 'Mp': res4['x'][0], 'wp': res4['x'][1],
                 'nll': res4['fval'], 'nllsd': res4['fsd'], 'success': res4['success'], 'func_count': res4['func_count']})
            Rslts = pd.concat([Rslts, new_row], ignore_index=True)
            print(f"Model 4 nll={res4['fval']}", end='\t')


            # Model 4b, with the denominator values independent from the numerator
            def noisy_object(x):
                nll = neg_ll_indv4b(x, dat)
                return nll


            LB = np.array([0, -1])
            UB = np.array([1000, 1])
            PLB = np.array([1.4, 0.1])
            PUB = np.array([100, .5])
            x0 = np.array([np.random.uniform(PLB[i], PUB[i]) for i in range(2)])
            bads = BADS(noisy_object, x0, LB, UB, PLB, PUB, options=options)
            res = bads.optimize()
            output = join(svdir, AnalysName, 'Model4b', f'PyBADS_{sublist[subj]}.npy')
            np.save(output, res)
            new_row = pd.DataFrame(
                {'subID': sublist[subj], 'Model': '4b', 'eta': [1], 'Mp': res['x'][0], 'wp': res['x'][1],
                 'nll': res['fval'], 'nllsd': res['fsd'], 'success': res['success'],
                 'func_count': res['func_count']})
            Rslts = pd.concat([Rslts, new_row], ignore_index=True)
            print(f"Model 4b nll={res['fval']}", end='\t')

            subj += 1
Rslts.to_csv(join(svdir, AnalysName, 'Rslts_PyBADS_Pro.txt'), sep='\t', index=False)

