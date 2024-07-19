library(rstan)
setwd("/Users/bs3667/Noise/UnusedCodeorForFutureAnalyses/modelfit")
# Simulate the data
set.seed(123)
n_trials <- 1000
utility_apple <- 5.3
utility_orange <- 3.5
beta <- utility_apple - utility_orange

# Simulate trial-level randomness (epsilon)
sigma_true <- 1  # true standard deviation of the trial-level random effect
epsilon_true <- rnorm(n_trials, mean = 0, sd = sigma_true)

# Simulate choices based on the true model
prob_apple <- 1 / (1 + exp(-(beta + epsilon_true)))
choices <- rbinom(n_trials, 1, prob_apple)

df <- data.frame(trial = 1:n_trials, choice = choices)

# Prepare data for Stan
stan_data <- list(
  N = nrow(df),
  y = df$choice
)

# Compile and fit the Stan model
# stan_model <- stan_model(file = 'choice_model_with_latent.stan')
stan_model <- stan_model(file = 'choice_model.stan')

# Fit the model
fit <- sampling(stan_model, data = stan_data, iter = 2000, warmup = 1000, chains = 4, seed = 123)

# Print the results
print(fit)
