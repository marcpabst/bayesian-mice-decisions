# This script uses `cmdstanr` to run the Stan model. Using R gives us access to a wider variety of tools, including those included in the `bayesplot` package.

# load stuff
library(BayesHMM)
library(bayesplot)
library(tidyverse)
library(bayestestR)
library(ggplot2)
library(abind)
library(cmdstanr)

# load data from csv
df <- read.csv("./data/ashwood.csv")


x <- df %>% group_by(session) %>%
  group_split() %>% 
  lapply(select, "stimulus", "bias") %>%
  lapply(as.matrix) %>%
  abind(along=1)

y <- df %>% group_by(session) %>%
  mutate(choice = 1 - choice) %>%
  group_split() %>% 
  lapply(select, "choice") %>%
  lapply(as.matrix) %>%
  abind(along=1)

T <- df %>% group_by(session) %>%
  count() %>%
  ungroup() %>%
  select(n) %>%
  as.matrix() %>%
  as.vector()


# specifiy input data for model
data <- list(
  x = x,
  y = drop(y),
  T = T,
  K = 3,
  R = 1,
  M = 4,
  N = length(T),
  I = sum(T)
)

# the model
model <- cmdstan_model("./stan-models/glm-hmm.stan")

# fit model
fit <- model$sample(
  data = data,            # named list of data
  chains = 1,             # number of Markov chains
  refresh = 5,             # print progress every 5 iterations
  iter_warmup = 1000,
  iter_sampling = 1000
  )
 
# load the posterior samples
stanfit <- rstan::read_stan_csv(fit$output_files())

# plot the posterior samples
mcmc_areas_ridges(stanfit, regex_pars = "betas")

# extract posterior predictive samples
ypred <- rstan::extract(stanfit, "ypred")

# plot the posterior predictive samples
ppc_bars(
  drop(y),
  ypred$ypred
)
