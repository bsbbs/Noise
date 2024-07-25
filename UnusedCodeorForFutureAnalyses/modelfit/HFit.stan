data {
  int<lower=1> maxNtrial;
  int<lower=1> Nsubj; // number of subjects
  int<lower=1> NTrial[Nsubj]; // number of trials per subject
  matrix[Nsubj,maxNtrial] V1;
  matrix[Nsubj,maxNtrial] V2;
  matrix[Nsubj,maxNtrial] V3;
  matrix[Nsubj,maxNtrial] sdV1;
  matrix[Nsubj,maxNtrial] sdV2;
  matrix[Nsubj,maxNtrial] sdV3;
  matrix[Nsubj,maxNtrial] chosenItem;
}
parameters {
  // hyper level mean, intermedium variable
  //real mu_M_pr; // reference point
  //real mu_w_pr; // lateral inhibition weight
  real mu_EN_pr; // early noise scaling parameter
  real mu_LN_pr; // late noise
  // hyper level sd, intermedium variable 
  //real<lower=0> sd_M;
  //real<lower=0> sd_w;
  real<lower=0> sd_EN;
  real<lower=0> sd_LN;
  
  // subjective-level raw parameters, declare as vectors for vectorizing
  //vector[Nsubj] M_pr; 
  //vector[Nsubj] w_pr;
  vector[Nsubj] EN_pr; 
  vector[Nsubj] LN_pr;
  
  vector<lower=0>[Nsubj] sigmasubj;
}
transformed parameters {
  //vector<lower=0,upper=90>[Nsubj] M;
  //vector<lower=0,upper=1>[Nsubj] w;
  vector<lower=.2,upper=2>[Nsubj] EN;
  vector<lower=0,upper=1>[Nsubj] LN;
  
  for (subj in 1:Nsubj) {
    EN[subj] = Phi_approx(mu_EN_pr + sd_EN * EN_pr[subj]) * 1.8 + .2;
    LN[subj] = Phi_approx(mu_LN_pr + sd_LN * LN_pr[subj]);
  }
}
model {
  // hyperparameters
  mu_EN_pr ~ normal(0, 1);
  mu_LN_pr ~ normal(0, 1);
  sd_EN  ~ cauchy(0, 5);
  sd_LN  ~ cauchy(0, 5);
  
  // individual parameters
  EN_pr ~ normal(0, 1);
  LN_pr ~ normal(0, 1);
  p = V1/
  SV1 = V1 + normal(V1, sdV2)
  SV2 =
  SV3 = 
  sigmasubj ~ cauchy(0, 5);
  for (subj in 1:Nsubj)
  {
    vhnorm[subj,1:NTrial[subj]] ~ normal(b1[subj]*Fairness[subj,1:NTrial[subj]] + b2[subj]*FairIntrcpt[subj,1:NTrial[subj]], sigmasubj[subj]);
  }
}

generated quantities {
  vector[Nsubj] log_lik;
  real LL_all;
  real mu_b1;
  real mu_b2;
  
  mu_b1 = (Phi_approx(mu_b1_pr) - .5)*20;
  mu_b2 = (Phi_approx(mu_b2_pr) - .5)*20;
  for (subj in 1:Nsubj){
    log_lik[subj] = normal_lpdf(vhnorm[subj,1:NTrial[subj]] | b1[subj]*Fairness[subj,1:NTrial[subj]] + b2[subj]*FairIntrcpt[subj,1:NTrial[subj]], sigmasubj[subj]);
  }
  LL_all=sum(log_lik);
}
