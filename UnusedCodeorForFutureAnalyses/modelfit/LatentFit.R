library(rstan)
setwd("C:/Users/Bo/Documents/GitHub/Noise/UnusedCodeorForFutureAnalyses/modelfit")
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
fit <- sampling(stan_model, data = stan_data, iter = 4000, warmup = 2000, chains = 4, seed = 123)

# Print the results
print(fit)






# Check convergence diagnostics
print(fit, pars = c("sigma"), probs = c(0.025, 0.5, 0.975))

# Trace plots for key parameters
traceplot(fit, pars = c("sigma"))
traceplot(fit, pars = c("z[1]"))
require(bayesplot)
mcmc_dens(fit, pars = c("sigma"))
mcmc_dens(fit, pars = c("z[1]", "z[2]", "z[3]"))
mcmc_dens(fit, pars = c("en[1]"))


# Posterior predictive checks
posterior_samples <- extract(fit)
sigma_samples <- posterior_samples$sigma
z_samples <- posterior_samples$z

# Summarize the results
sigma_hat <- mean(sigma_samples)
z_hat <- apply(z_samples, 2, mean)

# Print the estimated parameters
print(paste("Estimated sigma (trial-level SD): ", sigma_hat))
# Extract log-likelihood
log_lik <- extract(fit)$log_lik

# Summarize the log-likelihood
log_lik_mean <- apply(log_lik, 2, mean)
log_lik_sum <- sum(log_lik_mean)

print(paste("Sum of log-likelihood: ", log_lik_sum))


# Predict choices using the estimated alpha and epsilon
predicted_prob <- 1 / (1 + exp(-(1 + sigma_hat*z_hat)))


predicted_choices <- ifelse(predicted_prob > 0.5, 1, 0)

# Compare with actual choices
comparison <- data.frame(
  trial = df$trial,
  actual_choice = df$choice,
  predicted_choice = predicted_choices,
  predicted_prob = predicted_prob
)

# Calculate prediction accuracy
accuracy <- mean(comparison$actual_choice == comparison$predicted_choice)
print(paste("Prediction Accuracy: ", accuracy))

