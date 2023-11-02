# Visualize
rslts <- read.table(file = '/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults/ModelFitting/Rslts.txt', header = TRUE, sep = '\t')
rslts$sigma <- gsub("\\[|\\]", "", rslts$sigma)
rslts$sigma <- as.numeric(rslts$sigma)

library(tidyverse)
widedf <- pivot_wider(
  data = rslts,
  id_cols = subID,
  names_from = Model,
  values_from = nll)

widedf['dll'] <- widedf['2'] - widedf['1']
hist(widedf['dll']$dll)
hist(widedf['dll']$dll, breaks = c(-175, seq(-10,10,1)), xlim = c(-10,10), xlab = expression(paste(Delta, "nLL")), main = 'Model 2 - Model 1')
abline(v=0, lty = 2, col = 8)
dev.copy(pdf,'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults/ModelFitting/Mdl2vsMdl1_nLL.pdf', height=4, width=5, bg = 'transparent')
dev.off()
mean(widedf['dll']$dll, na.rm = TRUE)
# -2.897658
mean(widedf['dll']$dll[widedf['dll']$dll>-150], na.rm = TRUE)
# -0.07830976

rsltwd <- reshape(rslts, idvar = "subID", timevar = "Model", direction = "wide")
rsltwd$dsgm <- rsltwd$sigma.2 - rsltwd$sigma.1
rsltwd$dnll <- rsltwd$nll.2 - rsltwd$nll.1
rsltwd <- rsltwd[!is.na(rsltwd$dnll),]
hist(rsltwd$dsgm[rsltwd$dsgm > -100], breaks = seq(-30,60,.2), xlim = c(-1,1), xlab = expression(paste(Delta, sigma)), main = 'Model 2 - Model 1')
abline(v=0, lty = 2, col = 2)
dev.copy(pdf,'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults/ModelFitting/Mdl2vsMdl1_sigma.pdf', height=4, width=5, bg = 'transparent')
dev.off()
hist(rsltwd$dsgm, xlab = expression(paste(Delta, sigma)), main = 'Model 2 - Model 1')
abline(v=0, lty = 2, col = 2)
dev.copy(pdf,'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults/ModelFitting/Mdl2vsMdl1_sigma_.pdf', height=4, width=5, bg = 'transparent')
dev.off()
mean(rsltwd$dsgm)

mdl4 <- read.table(file = '/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults/ModelFitting/M4/GridSearch_Rslts.txt', header = TRUE, sep = '\t')
df <- merge(mdl4, rsltwd, by = c('subID'))
hist(df$nll - df$nll.2, main = 'Model 4 - Model 2', xlab = expression(paste(Delta, "nLL")))
abline(v=0, lty = 2, col = 2)
dev.copy(pdf,'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults/ModelFitting/Mdl4vsMdl2_nLL.pdf', height=4, width=5, bg = 'transparent')
dev.off()

mean(df$nll - df$nll.2)
hist(df$w, main = 'Model 4', 100, xlab = expression(paste("w")))
dev.copy(pdf,'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults/ModelFitting/Mdl4_w.pdf', height=4, width=5, bg = 'transparent')
dev.off()
mean(df$w)

hist(df$M, main = 'Model 4', 100, xlab = expression(paste("M")))
dev.copy(pdf,'/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults/ModelFitting/Mdl4_M.pdf', height=4, width=5, bg = 'transparent')
dev.off()
mean(df$M)


bads <- read.table(file = '/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/pyResults/ModelFitting/Rslts_PyBADS.txt', header = TRUE, sep = '\t')
badswd <- reshape(bads, idvar = "subID", timevar = "Model", direction = "wide")

badswd$dnLL <- badswd$nll.4 - badswd$nll.3

hist(badswd$dnLL)
mean(badswd$dnLL)

hist(badswd$Mp.3)
hist(badswd$Mp.4)
hist(badswd$wp.3)
hist(badswd$wp.4)
