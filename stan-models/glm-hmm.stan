functions {

  // forward algorithm
  vector[] forward(matrix log_b, matrix[] A, vector pi) {
   
    int K = dims(log_b)[1];
    int T = dims(log_b)[2];


    matrix[K,K] log_A[T];
    vector[K] log_pi;
    
    for (i in 1:K) {
      log_pi[i] = log(pi[i]);

      for (j in 1:K) {
        for (t in 1:T) {
          log_A[t][i,j] = log(A[t][i,j]);
        }
      }
    } 

    vector[T] logalpha[K];
    real acc[K];

    for(j in 1:K) {
      logalpha[j, 1] = log_pi[j] + log_b[j, 1];
    }

    for (t in 2:T) {
      for (j in 1:K) {
        for (i in 1:K) {
          acc[i] = logalpha[i, t-1] + log_A[t][i, j] + log_b[j, t];
        }
        logalpha[j, t] = log_sum_exp(acc);
      }
    }

    return logalpha;
  }

  // // Prior/posterior predictive
  // int[] zpredictive_rng(int K, int T, vector pi, vector[,] A) {
  //   int zpred[T];

  //   // Sample initial state
  //   zpred[1] = categorical_rng(pi);

  //   for(t in 2:T) {
  //     zpred[t] = categorical_rng(A[t - 1, zpred[t-1]]);
  //   }

  //   return zpred;
  // }
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
}

parameters {
    simplex[K] pi;   // initial state probabilities (one per sequence necessary?)
    vector[M] betas_x[K]; // per-state regression coefficients for obervation model

    simplex[K] baseA[K];  // per-state regression intercepts transition model
    //vector[P] betas_u[K]; // regression coefficients for transition model
}

transformed parameters {



  vector[N] log_like_sess;

  {
    int pos = 1;
    for(n in 1:N) {
      // compute forward values
      int T_ = T[n];
      matrix[T_, M] x_ = block(x, pos, 1, T_, M);

      matrix[K, K] A_[T_];

      int y_[T_] = y[pos:pos+T_-1];
      matrix[K, T_] logb_;

      for (t in 1:T_) {
        for (i in 1:K) {
          A_[t][i,] = to_row_vector(baseA[i,]);
          logb_[i][t] = bernoulli_logit_lpmf(y_[t] | x_[t] * betas_x[i]);
        }
      }

      //log_like_sess[n] = hmm_marginal(logb_, A_[1], pi);
      log_like_sess[n] = log_sum_exp( forward(logb_, A_, pi)[, T_] );

      pos = pos + T_;
    }
  }

}

model {
    for (k in 1:K) {
        baseA[k, ] ~ dirichlet([1,10]);
        // Gaussian priors for weights
        betas_x[k] ~ normal(0, 2);
    }
    
    for(n in 1:N) {
      target += log_like_sess[n];
    }
}

generated quantities {

  // int yhat[N, T];

  // {
  //   int pos = 1;
  //   for(n in 1:N) {
  //     hmm_latent_rng(logB, A, pi)

  //     zstar[pos:pos+T[n]-1,] =  most_likely_sequence(K, T, logpi, A, loglike[n]);
  //     log_like_sess[n] = log_sum_exp(calculate_forward_values(K, T[n], logpi, A, x[pos:pos+T[n]-1,], y[pos:pos+T[n]-1], betas_x)[, T[n]]); 
  //     pos = pos + T[n];
  //   }
  // }
  // //most likely state sequence for each observation sequence
  // for(n in 1:N) {
  //    zstar[n] =
  // }

}
