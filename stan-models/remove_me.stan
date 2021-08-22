functions {



// Forward algorithm log p(z_t = j | x_{1:t})
vector[] forward(int K, int T, vector logpi, vector[, ] logA, vector[] loglike) {
  vector[T] logalpha[K];
  real accumulator[K];

  for(j in 1:K) {
    logalpha[j, 1] = logpi[j] + loglike[j, 1];
  }

  for (t in 2:T) {
    for (j in 1:K) { // j = current (t)
    for (i in 1:K) { // i = previous (t-1)
    accumulator[i] = logalpha[i, t-1] + logA[t, i, j] + loglike[j, t];
    }
    logalpha[j, t] = log_sum_exp(accumulator);
    }
  }

  return logalpha;
}

  // Normalize
  vector normalize(real[] x) {
    return to_vector(x) / sum(x);
  }

  // Forward-backward algorithm log p(z_t = j | x_{1:T})
  vector[] forwardbackward(int K, int T, vector logpi, vector[,] logA, vector[] loglike, vector[] alpha) {
    vector[T] logbeta[K];
    vector[T] loggamma[K];
    vector[T] beta[K];
    vector[T] gamma[K];
    real accumulator[K];

    for (j in 1:K) {
      logbeta[j, T] = 1;
    }

    for (tforward in 0:(T-2)) {
      int t;
      t = T - tforward;

      for (j in 1:K) { // j = previous (t-1)
        for (i in 1:K) { // i = next (t)
          accumulator[i] = logbeta[i, t] + logA[t, j, i] + loglike[i, t];
        }
        logbeta[j, t-1] = log_sum_exp(accumulator);
      }
    }

    for (t in 1:T) {
      beta[, t] = to_array_1d(softmax(to_vector(logbeta[, t])));
    }

    for(t in 1:T) {
      loggamma[, t] = to_array_1d(to_vector(alpha[, t]) .* to_vector(beta[, t]));
    }

    for(t in 1:T) {
      gamma[, t] = to_array_1d(normalize(loggamma[, t]));
    }

    return gamma;
  } // Forward-ackward

// Viterbi arg_max ...
int[] MAPpath(int K, int T, vector logpi, vector[,] logA, vector[] loglike) {
  int zstar[T];
  real logp_zstar;
  int bpointer[T, K];
  real delta[T, K];

  for (j in 1:K)
    delta[1, j] = loglike[j, 1];

  for (t in 2:T) {
    for (j in 1:K) { // j = current (t)
    delta[t, j] = negative_infinity();
    for (i in 1:K) { // i = previous (t-1)
    real logp;
    logp = delta[t-1, i] + logA[t, i, j] + loglike[j, t];
    if (logp > delta[t, j]) {
      bpointer[t, j] = i;
      delta[t, j] = logp;
    }
    }
    }
  }

  logp_zstar = max(delta[T]);

  for (j in 1:K) {
    if (delta[T, j] == logp_zstar) {
      zstar[T] = j;
    }
  }

  for (t in 1:(T - 1)) {
    zstar[T - t] = bpointer[T - t + 1, zstar[T - t + 1]];
  }

  return zstar;
} // Viterbi

  // Prior/posterior predictive
  int[] zpredictive_rng(int K, int T, vector pi, vector[,] A) {
    int zpred[T];

    // Sample initial state
    zpred[1] = categorical_rng(pi);

    for(t in 2:T) {
      zpred[t] = categorical_rng(A[t - 1, zpred[t-1]]);
    }

    return zpred;
  }
    
}

data {
  int<lower=1> T;                   // number of observations (length)


    int<lower = 1> K; // number of hidden states
    int<lower = 1> R; // dimension of the observation vector
    
      int<lower = 1> P;     // number of transition model predictors
      matrix[T, P] u;       // transition model predictors
      
int<lower = 1> M; // number of predictors
matrix[T, M] x;   // predictors
int y[T, R];     // observations
}

transformed data {






}

parameters {


      simplex[K] pi;                    // initial state probabilities
      

vector[M] xBeta11;
vector[M] xBeta21;
vector[M] xBeta31;

}

transformed parameters {
  vector[T] loglike[K];


    vector[T] logalpha[K];
    vector[K] logpi;
    
vector[K] A[T, K];
		vector[K] logA[T, K];						 // transition logA[t, from, to]
      logpi = log(pi);
      
        for (t in 1:T) {
          for (i in 1:K) { // i = previous (t-1)


          }
          logA[t] = log(A[t]);
        }

  // Compute loglikelihood
  for (t in 1:T) {

loglike[1][t] = bernoulli_logit_lpmf(y[t] | x[t] * xBeta11);
loglike[2][t] = bernoulli_logit_lpmf(y[t] | x[t] * xBeta21);
loglike[3][t] = bernoulli_logit_lpmf(y[t] | x[t] * xBeta31);
  }

  // Compute target quantity


  logalpha = forward(K, T, logpi, logA, loglike);
  
}

model {

xBeta11 ~ normal(0, 2) ;
xBeta21 ~ normal(0, 2) ;
xBeta31 ~ normal(0, 2) ;


  target += log_sum_exp(logalpha[, T]);
  
}

generated quantities {
  matrix[T, R] ypred;
  int<lower=1, upper=K> zpred[T];



  vector[T] alpha[K];
  vector[T] gamma[K];
  int<lower=1, upper=K> zstar[T];
  for (t in 1:T)
    alpha[, t] = to_array_1d(softmax(to_vector(logalpha[, t])));
  gamma = forwardbackward(K, T, logpi, logA, loglike, alpha);
  zstar = MAPpath(K, T, logpi, logA, loglike);
  


  zpred = zpredictive_rng(K, T, pi, A);
  
  for(t in 1:T) {

if(zpred[t] == 1) ypred[t][1] = bernoulli_logit_rng(x[t] * xBeta11);
if(zpred[t] == 2) ypred[t][1] = bernoulli_logit_rng(x[t] * xBeta21);
if(zpred[t] == 3) ypred[t][1] = bernoulli_logit_rng(x[t] * xBeta31);
  }
}
