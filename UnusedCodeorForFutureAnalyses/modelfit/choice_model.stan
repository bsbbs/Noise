data {
  int<lower=0> N;            // number of trials
  int<lower=0, upper=1> y[N]; // choice data (1 for apple, 0 for orange)
}

parameters {
  real alpha;       // intercept (bias for choosing apple)
  real<lower=0> sigma; // standard deviation of trial-level random effect
  vector[N] z;      // trial-level random effects
}

transformed parameters {
  vector[N] eta;
  for (n in 1:N)
    eta[n] = alpha + sigma * z[n];
}

model {
  alpha ~ normal(0, 1);  // prior for intercept
  sigma ~ normal(0, 1);  // prior for sigma
  z ~ normal(0, 1);      // prior for trial-level random effects
  
  y ~ bernoulli_logit(eta);  // likelihood
}
