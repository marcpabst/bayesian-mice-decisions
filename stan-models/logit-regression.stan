data {
    int<lower=0> I; // total number of observations
    int<lower=0> M; // number of predictors observations

    matrix[I, M] x; // predictors
    int<lower=0,upper=1> y[I]; // observations
}

parameters {
    vector[M] betas;
}

model {
    //for (t in 1:T) {
        target += bernoulli_logit_lpmf(y | x * betas);
    //}
}

generated quantities {
  vector[I] log_lik;
  vector[I] y_hat;

  for (i in 1:I) {
    log_lik[i] = bernoulli_logit_lpmf(y[i] | y[i] * betas);
    y_hat[i] = bernoulli_logit_rng(x[i] * betas);
  }
}
