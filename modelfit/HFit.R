# Hierachical fitting the behavior of noise project
require('rstan')
# define directories
datadir <- "/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/myData"
dumpdir <- "/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject/Modelfit/Hfit/objects"
plotdir <- "/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject/Modelfit/Hfit/plot"
# load data
mydat <- read.csv('/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/myData/TrnsfrmData.csv', header = TRUE)

# define 
filename <- file.path(dumpdir, sprintf('.RData'))
if (!file.exists(filename)){
  TrialIDs <- aggregate(trial ~ subNo, data = tbingdat, FUN = unique)
  NTrial <- aggregate(trial ~ subNo, data = tbingdat, FUN = length)
  
  maxNtrial <- max(NTrial$trial)
  Fairness <- matrix(rep(rep(0,maxNtrial),Nsubj),Nsubj,maxNtrial)
  FairIntrcpt <- matrix(rep(rep(0,maxNtrial),Nsubj),Nsubj,maxNtrial)
  vhnorm <- matrix(rep(rep(0,maxNtrial),Nsubj),Nsubj,maxNtrial)
  for (subj in 1:Nsubj)
  {
    Fairness[subj,1:NTrial$trial[NTrial$subNo == gsublist[subj]]] <- tbingdat$Fairness[tbingdat$subNo == gsublist[subj]]
    FairIntrcpt[subj,1:NTrial$trial[NTrial$subNo == gsublist[subj]]] <- tbingdat$FairIntrcpt[tbingdat$subNo == gsublist[subj]]
    vhnorm[subj,1:NTrial$trial[NTrial$subNo == gsublist[subj]]] <- tbingdat$vhnorm[tbingdat$subNo == gsublist[subj]]
  }
  stanobj <- list(maxNtrial = maxNtrial, Nsubj = Nsubj, NTrial = NTrial$trial, Fairness = Fairness, FairIntrcpt = FairIntrcpt, vhnorm = vhnorm)
  