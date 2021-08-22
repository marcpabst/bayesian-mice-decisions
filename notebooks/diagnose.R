# This is a script to diagnose the 3-state GLM-HMM.   

library(BayesHMM)
library(bayesplot)
library(tidyverse)
library(bayestestR)
library(ggplot2)
library(abind)
library(cmdstanr)


# load diag.csv
df <- read.csv("./data/diag.csv")



x <- df %>% group_by(session_start_time) %>%
  group_split() %>% 
  lapply(select, "intercept", "stim_contrast", "stim_contrast_minus_1", "response_cw_minus_1") %>%
  lapply(as.matrix) %>%
  abind(along=1)

y <- df %>% group_by(session_start_time) %>%
  mutate(choice = 1 - as.integer(as.logical(response_cw))) %>%
  group_split() %>% 
  lapply(select, "choice") %>%
  lapply(as.matrix) %>%
  abind(along=1)

T <- df %>% group_by(session_start_time) %>%
  count() %>%
  ungroup() %>%
  select(n) %>%
  as.matrix() %>%
  as.vector()


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
  M = 2,
  N = length(T),
  I = sum(T)
)


# the model
model <- cmdstan_model("./stan-models/glm-hmm.stan")

# fit model
fit <- model$sample(
  data = data,            # named list of data
  chains = 4,             # number of Markov chains
  refresh = 5,             # print progress every 5 iterations
  iter_warmup = 1000,
  iter_sampling = 1000
  )
 



stanfit <- rstan::read_stan_csv(fit$output_files())

library(ggplot2)
library(tidybayes)

mcmc_areas_ridges(stanfit, regex_pars = "betas")

