data {
  int<lower=0> N;            // number of trials
  int<lower=0, upper=1> y[N]; // choice data (1 for apple, 0 for orange)
}

parameters {
  // real alpha;                // intercept (bias for choosing apple)
  real<lower=0> sigma;       // standard deviation of trial-level random effect
  vector[N] epsilon;         // trial-level random effects
}

model {
  // Priors
  // alpha ~ normal(0, 1);      
  sigma ~ cauchy(0, 1);      
  epsilon ~ normal(0, sigma);
  
  // Likelihood
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(1 + epsilon[n]);
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | 1 + epsilon[n]);
  }
}
