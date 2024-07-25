data {
  int<lower=0> N;            // number of trials
  int<lower=0, upper=1> y[N]; // choice data (1 for apple, 0 for orange)
}

parameters {
  // real alpha;       // intercept (bias for choosing apple)
  real<lower=0> sigma; // standard deviation of trial-level random effect
  // vector[N] eta;      // trial-level random effects
}

transformed parameters {
  vector[N] eta;
  eta ~ normal(0, sigma);
}

model {
  // alpha ~ normal(0, 1);  // prior for intercept
  sigma ~ normal(0, 1);  // prior for sigma
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(1 + eta[n]);  // likelihood
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] eta;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | 1 + eta[n]);
  }
}
