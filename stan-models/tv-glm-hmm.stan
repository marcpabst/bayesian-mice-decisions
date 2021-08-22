functions {
  // forward algorithm
  vector[] forward(int K, int T, vector logpi, vector[,] logA, vector[] loglike) {
    vector[T] logalpha[K];
    real acc[K];

    for(j in 1:K) {
      logalpha[j, 1] = logpi[j] + loglike[j, 1];
    }

    for (t in 2:T) {
      for (j in 1:K) {
        for (i in 1:K) {
          acc[i] = logalpha[i, t-1] + logA[t, i, j] + loglike[j, t];
        }
        logalpha[j, t] = log_sum_exp(acc);
      }
    }

    return logalpha;
  } 


// calculates the forward values alpha_i,j(t)
  vector[] calculate_forward_values(int K, int T, vector logpi,  vector[] baseP, matrix x, matrix u, int[] y, vector[] betas_x, vector[] betas_u) {
    vector[T] logalpha[K];
    vector[T] logb[K];
    vector[K] logA[T, K];  // time-varying transition matrix (on the log scale)

    vector[K] temp;

    for (t in 1:T) {
        for (i in 1:K) {
            // for (j in 1:K) {
            //     temp[j] = baseP[i,j] + (u[t] * betas_u[j]);
            // }

            logA[t][i] = baseP[i];
            logb[i][t] = bernoulli_logit_lpmf(y[t] | x[t] * betas_x[i]);
        }

    }
    // calculate forward values alpha_i,j(t) for all t
    logalpha = forward(K, T, logpi, logA, logb);
    return logalpha;
  }

}

data {
    // sequences
    int<lower = 1> N;   // number of sequences
    int T[N];           // length of each sequence

    int<lower = 1> K;   // number of hidden states

    // observation model
    int<lower = 0> I;   // number of (flattend/total) obervations
    int<lower = 1> R;   // dimensionality of observations (NOT USED atm) 
    int<lower = 1> M;   // number of predictors for observations

    int y[I];        // observations
    matrix[I, M] x;     // predictors for observations

    // transition model
    int<lower = 1> P;   // number of predictors for latent transition probabilities
    matrix[I, P] u;     // predictors for transitions

    
} 

parameters {
    simplex[K] pi;   // initial state probabilities (one per sequence necessary?)
    vector[M] betas_x[K]; // per-state regression coefficients for obervation model

    simplex[K] baseP[K];  // per-state regression intercepts transition model
    vector[P] betas_u[K]; // regression coefficients for transition model
}

transformed parameters {
    
    vector[K] logpi;
    vector[N] log_like_sess;

    for (k in 1:K) {
        logpi[k] = log(pi[k]);
    }    
    {
      int pos = 1;
      for(n in 1:N) {
        log_like_sess[n] = log_sum_exp(calculate_forward_values(K, T[n], logpi, baseP, x[pos:pos+T[n]-1,],  u[pos:pos+T[n]-1,], y[pos:pos+T[n]-1], betas_x, betas_u)[, T[n]]); 
        pos = pos + T[n];
      }
    }

}

model {
    int pos;
    pos = 1;

    for (k in 1:K) {
        baseP[k, ] ~ dirichlet(rep_vector(1, K));
        // Gaussian priors for weights
        betas_x[k] ~ normal(0, 4);
        betas_u[k] ~ normal(0, 4);
    }
    
    for(n in 1:N) {
      target += log_like_sess[n];
      pos = pos + T[n];
    }
}

generated quantities {
  // int zstar[N, T];

  // // most likely state sequence for each observation sequence
  // for(n in 1:N) {
  //   zstar[n] = most_likely_sequence(K, T, logpi[n], logA, loglike[n]);
  // }
}
