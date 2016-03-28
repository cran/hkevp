
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;



/* multiGaussRand_cpp
 * Generates one realization of the multivariate Gaussian distribution.
 * 
 * nrep: number of simulations.
 * mu: mean vector.
 * sigma: covariance matrix.
 */
arma::mat multiGaussRand_cpp(arma::vec const& mu, arma::mat const& Sigma)
{
  arma::mat cholesky = chol(Sigma);
  arma::mat result = cholesky.t()*arma::randn(mu.size()) + mu;
  return result.t();
}


/* truncatedNormRand_cpp
 * Generates a random variable from the truncated normal distribution on [0,1].
 * 
 * mean: mean of the truncated normal distribution.
 * sd: standard deviation of the truncated normal distribution.
 */

double truncatedNormRand_cpp(double const& mean, double const& sd)
{
  return R::qnorm(R::runif(R::pnorm(0, mean, sd, 1, 0),
                           R::pnorm(1, mean, sd, 1, 0)), mean, sd, 1, 0);
}


/* truncatedNormDensity_cpp
 * Probability density function of the truncated normal distribution on [0,1].
 * 
 * x: where the function is evaluated.
 * mean: mean of the truncated normal distribution.
 * sd: standard deviation of the truncated normal distribution.
 */
double truncatedNormDensity_cpp(double const& x, double const& mean, double const& sd)
{
  return R::dnorm(x, mean, sd, 1)-log(R::pnorm(1, mean, sd, 1, 0)-R::pnorm(0, mean, sd, 1, 0));
}



/* positiveStableDensity_cpp
 * Probability density function (in log) of the positive stable distribution with characteristic exponent alpha.
 * 
 * A: random effect of the positive stable distribution.
 * B: auxiliary variable (see Stephenson 2009).
 * alpha: characterstic exponent.
 */
double positiveStableDensity_cpp(double const& A, double const& B, double const& alpha)
{
  double psi = arma::datum::pi*B;
  double C = pow(sin(alpha*psi)/sin(psi), (1/(1-alpha))) * sin((1-alpha)*psi)/sin(alpha*psi);
  double logDensity = log(alpha) - log(1-alpha) - (1/(1-alpha))*log(A) + log(C) - C*pow(A, -alpha/(1-alpha));
  
  return logDensity;
}


/* positiveStableDensity_forMatrices_cpp
 * Probability density function (in log) of the positive stable distribution with characteristic exponent alpha, where variables A and B are matrices.
 * 
 * A: random effect of the positive stable distribution (matrix).
 * B: auxiliary variable (see Stephenson 2009, matrix).
 * alpha: characteristic exponent.
 */
double positiveStableDensity_forMatrices_cpp(arma::mat const& A, arma::mat const& B, double const& alpha)
{
  arma::mat psi = arma::datum::pi*B;
  arma::mat C = pow(sin(alpha*psi)/sin(psi), (1/(1-alpha))) % sin((1-alpha)*psi)/sin(alpha*psi);
  arma::mat logDensity = log(alpha) - log(1-alpha) - (1/(1-alpha))*log(A) + log(C) - C%pow(A, -alpha/(1-alpha));
  
  return sum(sum(logDensity));
}





/* kernelMatrix_cpp
 * Computes the matrix of kernels Omega with given distance matrix and bandwidth parameter.
 * 
 * dsk: distance matrix between sites and knots.
 * tau: bandwidth parameter.
 */
arma::mat kernelMatrix_cpp(arma::mat const& dsk, double const& tau)
{
  arma::mat result = exp(-pow(dsk,2) / (2*pow(tau,2)) ).t();
  result.each_row() /= sum(result);
  return result;
}



/* spatialDependenceProcess_cpp
 * Builds the spatial dependence process theta of the HKEVP.
 * 
 * dsk: distance matrix between sites and knots.
 * A: positive stable random effect .
 * alpha: dependence parameter.
 * tau: bandwidth parameter.
 */
arma::mat spatialDependenceProcess_cpp(arma::mat const& dsk, arma::mat const& A, double const& alpha, double const& tau)
{
  arma::mat omega = kernelMatrix_cpp(dsk, tau);
  arma::mat theta = arma::mat(A.n_cols, dsk.n_rows);
  theta = pow( A*pow(omega, 1/alpha) , alpha);
  return theta;
}




/* logLikelihoodCore_cpp
 * Computes the core of the log-likelihood : (1+xi/sig * (y-mu))_+ ^(-1/alpha*xi)
 * 
 * y: observed process.
 * gev: matrix of GEV parameters (mu, log(sigma), xi).
 * alpha: dependence parameter of HKEVP.
 */
arma::mat logLikelihoodCore_cpp(arma::mat const& y, arma::mat const& gev, double const& alpha)
{
  arma::mat result = y;
  result.each_row() -= gev.col(0).t();
  result.each_row() %= gev.col(2).t()/exp(gev.col(1)).t();
  result ++;
  for (int i=0; i<gev.n_rows; i++)
  {
    result.col(i) = pow(result.col(i), -1/(alpha*gev(i,2)));
  }
  
  return arma::is_finite(result)? result : arma::zeros<arma::mat>(y.n_rows, y.n_cols);
}



/* logLikelihood_cpp
 * Computes the log likelihood of the HKEVP for an observed process y.
 * 
 * 
 * y: observed process.
 * gev: matrix of GEV parameters (mu, log(sigma), xi).
 * alpha: dependence parameter of HKEVP.
 * theta: spatial dependence process.
 * nas: matrix of NAs.
 */
double logLikelihood_cpp(arma::mat const& y, arma::mat const& gev, double const& alpha, arma::mat const& theta, arma::mat const& nas)
{
  arma::vec llik_bySites = arma::zeros<arma::mat>(y.n_cols);
  arma::vec coreGEV = arma::zeros<arma::mat>(y.n_rows);
  
  for (int i=0; i<y.n_cols; i++)
  {
    coreGEV = pow(1+gev(i,2)*(y.col(i)-gev(i,0))/exp(gev(i,1)),-1/(alpha*gev(i,2)));
    llik_bySites(i) = is_finite(coreGEV) ?
    sum( (-log(alpha*exp(gev(i,1))) + log(theta.col(i))/alpha + (alpha*gev(i,2)+1)*log(coreGEV) - pow(theta.col(i),1/alpha)%coreGEV)%nas.col(i) ) :
      -arma::datum::inf;
  }
  
  return sum(llik_bySites);
}






/* mcmc_hkevp
 * MCMC procedure that fits the HKEVP on observations Y.
 * 
 * Y: observed process.
 * sites: spatial positions where Y is observed.
 * spatial_covariates: spatial covariates for GEV parameters.
 * knots: spatial positions of the knots.
 * dsk: distance matrix between sites and knots.
 * dss: distance matrix between sites.
 * niter: number of iterations before the algorithm stops.
 * nburn: number of first iterations to burn.
 * nthin: size of thinning.
 * trace: size of tracing display.
 * gev_vary: are gev parameters spatially varying?
 * gev_init: initial state of GEV parameters.
 * alpha_init: initial state of alpha.
 * tau_init: initial state of tau.
 * A_init: initial state of A.
 * B_init: initial state of B.
 * sills_init: initial state of sills.
 * ranges_init: initial state of ranges.
 * constant_gev_prior: prior hyperparameters for spatially-constant GEV parameters.
 * alpha_prior: prior hyperparameters for alpha.
 * tau_prior: prior hyperparameters for tau.
 * beta_variance_prior: prior hyperparameter for variance of Beta's.
 * sill_prior: prior hyperparameters for sills.
 * range_prior: prior hyperparameters for ranges.
 * gev_random_walk: random walk standard deviation for GEV parameters.
 * range_random_walk: random walk standard deviation for ranges.
 * tau_random_walk: random walk standard deviation for tau.
 * alpha_random_walk: random walk standard deviation for alpha.
 * A_random_walk: random walk standard deviation for A.
 * B_random_walk: random walk standard deviation for B.
 * quiet: display or not.
 * latent_processes_correlation_type: type of spatial correlation (latent processes).
 * nas: matrix of NAs.
 */
// [[Rcpp::export]]
Rcpp::List MCMC(arma::mat const& Y, 
                arma::mat const& sites,
                arma::mat const& spatial_covariates,
                arma::mat const& knots,
                arma::mat const& dsk,
                arma::mat const& dss,
                int const& niter,
                int const& nburn,
                int const& nthin,
                int const& trace,
                arma::vec const& gev_vary,
                arma::mat const& gev_init,
                double const& alpha_init,
                double const& tau_init,
                double const& A_init,
                double const& B_init,
                double const& sills_init,
                double const& ranges_init,
                arma::mat const& constant_gev_prior,
                arma::vec const& alpha_prior,
                arma::vec const& tau_prior,
                double const& beta_variance_prior,
                arma::mat const& sill_prior,
                arma::mat const& range_prior,
                arma::vec const& gev_random_walk,
                arma::vec const& range_random_walk,
                double const& tau_random_walk,
                double const& alpha_random_walk,
                double const& A_random_walk,
                double const& B_random_walk,
                bool const& quiet,
                std::string const& latent_processes_correlation_type,
                arma::mat const& nas)
{
  /*  * * * Initialisation * * *  */
  
  // General variables:
  int n_sites = sites.n_rows;
  int n_knots = knots.n_rows;
  int n_years = Y.n_rows;
  int n_covar = spatial_covariates.n_cols;
  
  
  // Maximum distance between sites -- used in the Beta prior distribution for bandwidth TAU and GEV-range
  double distMax = max(max(dss));
  
  // Priors for beta:
  arma::mat BETA_covar_prior = arma::eye<arma::mat>(n_covar, n_covar) * beta_variance_prior;
  
  
  
  
  // ----- Declaration of Markov chains variables
  // GEV params:
  arma::cube GEV_chain = arma::zeros<arma::cube>(n_sites, 3, niter-nburn);
  arma::mat sills_chain = arma::ones<arma::mat>(niter-nburn, 3);
  arma::mat ranges_chain = arma::ones<arma::mat>(niter-nburn, 3);
  arma::cube BETA_chain = arma::ones<arma::cube>(niter-nburn, n_covar, 3);
  
  // Spatial parameters:
  arma::mat GEVmeanVectors = arma::ones<arma::mat>(n_sites, 3);
  arma::cube GEVcovarMatrices = arma::ones<arma::cube>(n_sites, n_sites, 3);
  
  // Dependence parameters:
  arma::vec alpha_chain = arma::zeros<arma::vec>(niter-nburn);
  arma::vec tau_chain = arma::zeros<arma::vec>(niter-nburn);
  arma::mat A = arma::zeros<arma::mat>(n_years, n_knots);
  arma::mat B = arma::zeros<arma::mat>(n_years, n_knots);
  arma::cube A_chain = arma::ones<arma::cube>(n_years, n_knots, niter-nburn);
  
  // Log-likelihood:
  arma::vec llik_chain = arma::zeros<arma::vec>(niter-nburn);
  int llik_count = 0;
  
  
  
  // ----- Initialisation of the Markov chains
  // GEV parameters:
  GEV_chain.slice(0) = gev_init;
  
  // Spatial parameters:
  std::string corr_expo("expo"), corr_gauss("gauss"), corr_mat32("mat32"), corr_mat52("mat52");
  sills_chain.row(0) *= sills_init;
  ranges_chain.row(0) *= ranges_init;
  BETA_chain(0,0,0) = mean(GEV_chain.slice(0).col(0)); // Beta_0 = mean(mu)
  BETA_chain(0,0,1) = mean(GEV_chain.slice(0).col(1)); // Beta_0 = mean(gamma)
  BETA_chain(0,0,2) = mean(GEV_chain.slice(0).col(2)); // Beta_0 = mean(xi)
  GEVmeanVectors.col(0) = spatial_covariates * BETA_chain.slice(0).row(0).t();
  GEVmeanVectors.col(1) = spatial_covariates * BETA_chain.slice(1).row(0).t();
  GEVmeanVectors.col(2) = spatial_covariates * BETA_chain.slice(2).row(0).t();
  if (latent_processes_correlation_type==corr_expo) {
    GEVcovarMatrices.slice(0) = sills_init * exp(-dss/ranges_init);
    GEVcovarMatrices.slice(1) = sills_init * exp(-dss/ranges_init);
    GEVcovarMatrices.slice(2) = sills_init * exp(-dss/ranges_init);
  } else if (latent_processes_correlation_type==corr_gauss) {
    GEVcovarMatrices.slice(0) = sills_init * exp(-pow(dss/ranges_init, 2.0)/2);
    GEVcovarMatrices.slice(1) = sills_init * exp(-pow(dss/ranges_init, 2.0)/2);
    GEVcovarMatrices.slice(2) = sills_init * exp(-pow(dss/ranges_init, 2.0)/2);
  } else if (latent_processes_correlation_type==corr_mat32) {
    GEVcovarMatrices.slice(0) = sills_init * (1+sqrt(3)*dss/ranges_init) % exp(-sqrt(3)*dss/ranges_init);
    GEVcovarMatrices.slice(1) = sills_init * (1+sqrt(3)*dss/ranges_init) % exp(-sqrt(3)*dss/ranges_init);
    GEVcovarMatrices.slice(2) = sills_init * (1+sqrt(3)*dss/ranges_init) % exp(-sqrt(3)*dss/ranges_init);
  } else if (latent_processes_correlation_type==corr_mat52) {
    GEVcovarMatrices.slice(0) = sills_init * (1+dss/ranges_init*sqrt(5.0) + 5.0/3.0*pow(dss/ranges_init, 2.0)) % exp(-dss/ranges_init *sqrt(5.0));
    GEVcovarMatrices.slice(1) = sills_init * (1+dss/ranges_init*sqrt(5.0) + 5.0/3.0*pow(dss/ranges_init, 2.0)) % exp(-dss/ranges_init *sqrt(5.0));
    GEVcovarMatrices.slice(2) = sills_init * (1+dss/ranges_init*sqrt(5.0) + 5.0/3.0*pow(dss/ranges_init, 2.0)) % exp(-dss/ranges_init *sqrt(5.0));
  }
  
  
  // Dependence parameters:
  alpha_chain(0) = alpha_init;
  tau_chain(0) = tau_init;
  A.fill(A_init);
  B.fill(B_init);
  
  
  
  
  
  // ----- Other variables used in the loop
  // Current states:
  arma::mat GEV_current = GEV_chain.slice(0);
  double tau_current = tau_chain(0);
  double alpha_current = alpha_chain(0);
  double llik_current = llik_chain(0);
  arma::mat theta_current = spatialDependenceProcess_cpp(dsk, A, alpha_init, tau_init);
  arma::mat A_current = A;
  arma::mat B_current = B;
  arma::vec sills_current = sills_chain.row(0).t();
  arma::vec ranges_current = ranges_chain.row(0).t();
  arma::mat BETA_current = arma::zeros<arma::mat>(n_covar,3);
  BETA_current.col(0) = BETA_chain.slice(0).row(0).t();
  BETA_current.col(1) = BETA_chain.slice(1).row(0).t();
  BETA_current.col(2) = BETA_chain.slice(2).row(0).t();
  
  // Log-likelihood:
  llik_chain(0) = logLikelihood_cpp(Y, GEV_current, alpha_current, theta_current, nas); llik_count++;
  if (!arma::is_finite(llik_chain(0))) {stop("Bad initialisation: Null Likelihood!");}
  
  // Candidates:
  arma::mat GEV_candidate = GEV_current;
  double tau_candidate, alpha_candidate, llik_candidate;
  arma::mat A_candidate = A_current;
  arma::mat B_candidate = B_current;
  arma::mat theta_candidate = theta_current;
  double range_candidate;
  arma::mat covarMatrix_candidate(n_sites, n_sites);
  
  // Log-ratios:
  double GEV_logratio, tau_logratio, alpha_logratio, A_logratio, B_logratio, range_logratio;
  
  // Others
  arma::mat omega = kernelMatrix_cpp(dsk, tau_current).t();
  arma::mat GEV_covarMatrix_inverted(n_sites, n_sites);
  arma::vec meanVector(n_sites);
  arma::mat llik_core = logLikelihoodCore_cpp(Y, GEV_current, alpha_current);
  double llik_diff(0.0);
  arma::mat quadraticForm = arma::zeros<arma::mat>(n_covar, n_covar);
  arma::vec BETA_conjugate_meanVector(n_covar);
  arma::mat BETA_conjugate_covarianceMatrix(n_covar, n_covar);
  arma::vec sill_conjugate = arma::zeros<arma::mat>(2);
  
  
  
  // Timers
  arma::wall_clock timer;
  arma::vec A_timer = arma::zeros<arma::vec>(niter);
  arma::vec B_timer = arma::zeros<arma::vec>(niter);
  arma::vec alpha_timer = arma::zeros<arma::vec>(niter);
  arma::vec tau_timer = arma::zeros<arma::vec>(niter);
  arma::vec GEV_timer = arma::zeros<arma::vec>(niter);
  
  
  
  /*  * * * * * BEGINNING OF SAMPLE * * * * *  */
  for (int R=0; R<niter; R++)
  {
    for (int th=0; th<nthin; th++)
    {
      /*  * * * Random effect A * * *  */
      timer.tic();
      omega = kernelMatrix_cpp(dsk, tau_current).t();
      omega = pow(omega, 1/alpha_current).t();
      llik_core = logLikelihoodCore_cpp(Y, GEV_current, alpha_current);
      
      
      for (int t=0; t<n_years; t++)
      {
        for (int k=0; k<n_knots; k++)
        {
          // Candidate A
          A_candidate(t,k) = exp(R::rnorm(log(A_current(t,k)), A_random_walk));
          
          // Candidate THETA
          theta_candidate = theta_current;
          theta_candidate.row(t) = pow( pow(theta_current.row(t), 1/alpha_current)
                                          + (A_candidate(t,k)-A_current(t,k))*omega.row(k), alpha_current);
          
          // Log-ratio
          llik_diff = sum(( (log(theta_candidate.row(t))-log(theta_current.row(t)))/alpha_current
                              + (A_current(t,k)-A_candidate(t,k)) * (llik_core.row(t)%omega.row(k)) )%nas.row(t));
          A_logratio = llik_diff
            + positiveStableDensity_cpp(A_candidate(t,k), B_current(t,k), alpha_current)
            - positiveStableDensity_cpp(A_current(t,k), B_current(t,k), alpha_current)
            + log(A_candidate(t,k)) - log(A_current(t,k));
            
            // Acceptance test
            if ( (exp(A_logratio)>R::runif(0,1)) & (arma::is_finite(A_logratio)))
            {
              A_current(t,k) = A_candidate(t,k);
              theta_current.row(t) = theta_candidate.row(t);
            }
            
        }
      }
      
      llik_current = logLikelihood_cpp(Y, GEV_current, alpha_current, theta_current, nas); llik_count++;
      A_timer(R) = timer.toc();
      
      /*  * * * Auxiliary variable B * * *  */
      timer.tic();
      for (int k=0; k<n_knots; k++)
      {
        for (int t=0; t<n_years; t++)
        {
          // Candidate
          B_candidate(t,k) = truncatedNormRand_cpp(B_current(t,k), B_random_walk);
          
          // Log-ratio
          B_logratio = positiveStableDensity_cpp(A_current(t,k), B_candidate(t,k), alpha_current)
            - positiveStableDensity_cpp(A_current(t,k), B_current(t,k), alpha_current)
            + truncatedNormDensity_cpp(B_current(t,k), B_candidate(t,k), B_random_walk)
            - truncatedNormDensity_cpp(B_candidate(t,k), B_current(t,k), B_random_walk);
            
            // Acceptance test
            if ( (exp(B_logratio)>R::runif(0,1)) & (arma::is_finite(B_logratio)) )
            {
              B_current(t,k) = B_candidate(t,k);
            }
        }
      }
      B_timer(R) = timer.toc();
      
      
      
      
      /*  * * * ALPHA * * *  */
      timer.tic();
      
      // Candidates
      alpha_candidate = truncatedNormRand_cpp(alpha_current, alpha_random_walk);
      theta_candidate = spatialDependenceProcess_cpp(dsk, A_current, alpha_candidate, tau_current);
      llik_candidate = logLikelihood_cpp(Y, GEV_current, alpha_candidate, theta_candidate, nas); llik_count++;
      
      // log-ratio computation
      alpha_logratio = llik_candidate - llik_current
        + positiveStableDensity_forMatrices_cpp(A_current, B_current, alpha_candidate)
        - positiveStableDensity_forMatrices_cpp(A_current, B_current, alpha_current)
        + R::dbeta(alpha_candidate, alpha_prior(0), alpha_prior(1), 1)
        - R::dbeta(alpha_current, alpha_prior(0), alpha_prior(1), 1)
        + truncatedNormDensity_cpp(alpha_current, alpha_candidate, alpha_random_walk)
        - truncatedNormDensity_cpp(alpha_candidate, alpha_current, alpha_random_walk)
        ;
      
      // Acceptance test
      if ( (exp(alpha_logratio)>R::runif(0,1)) & (arma::is_finite(alpha_logratio)) )
      {
        alpha_current = alpha_candidate;
        theta_current = theta_candidate;
        llik_current = llik_candidate;
      }
      alpha_timer(R) = timer.toc();
      
      
      
      
      /*  * * * TAU * * *  */
      timer.tic();
      
      // Candidates
      tau_candidate = exp(R::rnorm(log(tau_current), tau_random_walk));
      theta_candidate = spatialDependenceProcess_cpp(dsk, A_current, alpha_current, tau_candidate);
      llik_candidate = logLikelihood_cpp(Y, GEV_current, alpha_current, theta_candidate, nas); llik_count++;
      
      // Log-ratio computation
      tau_logratio = llik_candidate-llik_current
        + R::dbeta(tau_candidate/(2*distMax), tau_prior(0), tau_prior(1), 1)
        - R::dbeta(tau_current/(2*distMax), tau_prior(0), tau_prior(1), 1)
        + log(tau_candidate) - log(tau_current)
        ;
      
      // Acceptance test
      if ( (exp(tau_logratio)>R::runif(0,1)) & (arma::is_finite(tau_logratio)) )
      {
        tau_current = tau_candidate;
        theta_current = theta_candidate;
        llik_current = llik_candidate;
      }
      tau_timer(R) = timer.toc();
      
      
      
      
      /*  * * *  GEV Parameters  * * *  */
      timer.tic();
      
      for (int n_gev=0; n_gev<3; n_gev++)
      {
        llik_current = logLikelihood_cpp(Y, GEV_current, alpha_current, theta_current, nas); llik_count++;
        
        // -------------------- IF the GEV parameters is VARYING (spatially) --------------------
        if (gev_vary(n_gev)==1) {
          
          GEV_covarMatrix_inverted = inv(GEVcovarMatrices.slice(n_gev));
          
          // ---------- STEP 1: Updating site by site
          for (int i=0; i<n_sites; i++)
          {
            // Candidates
            GEV_candidate = GEV_current;
            GEV_candidate(i,n_gev) = R::rnorm(GEV_current(i,n_gev), gev_random_walk(n_gev));
            llik_candidate = logLikelihood_cpp(Y, GEV_candidate, alpha_current, theta_current, nas); llik_count++; // Optimisable selon le paramètre regardé ?
            
            // Log-ratio
            GEV_logratio = llik_candidate-llik_current
              - as_scalar((GEV_candidate.col(n_gev)-GEVmeanVectors.col(n_gev)).t() * GEV_covarMatrix_inverted * (GEV_candidate.col(n_gev)-GEVmeanVectors.col(n_gev))/2)
              + as_scalar((GEV_current.col(n_gev)-GEVmeanVectors.col(n_gev)).t() * GEV_covarMatrix_inverted * (GEV_current.col(n_gev)-GEVmeanVectors.col(n_gev))/2)
              ;
            
            // Acceptance test
            if ( (exp(GEV_logratio)>R::runif(0,1)) & (arma::is_finite(GEV_logratio)) )
            {
              GEV_current(i,n_gev) = GEV_candidate(i,n_gev);
              llik_current = llik_candidate;
            }
          }
          
          
          
          // ---------- STEP 2: Updating the mean vector using normal conjugate prior
          
          // Conjugate distribution parameters
          BETA_conjugate_covarianceMatrix = inv(inv(BETA_covar_prior) + (spatial_covariates.t() * GEV_covarMatrix_inverted * spatial_covariates));
          BETA_conjugate_meanVector = BETA_conjugate_covarianceMatrix * (spatial_covariates.t() * GEV_covarMatrix_inverted * GEV_current.col(n_gev));
          
          // Regression parameters BETA
          BETA_current.col(n_gev) = multiGaussRand_cpp(BETA_conjugate_meanVector, BETA_conjugate_covarianceMatrix).t();
          
          // Mean vector of GEV
          GEVmeanVectors.col(n_gev) = spatial_covariates * BETA_current.col(n_gev);
          
          
          
          
          // ---------- STEP 3: Updating the sill using Gamma conjugate prior
          // Quadratic form used in the "b" parameter of the gamma prior
          quadraticForm = (GEV_current.col(n_gev)-GEVmeanVectors.col(n_gev)).t() * GEV_covarMatrix_inverted * (GEV_current.col(n_gev)-GEVmeanVectors.col(n_gev));
          
          // New sill
          sill_conjugate(0) = n_sites/2 + sill_prior(0,0);
          sill_conjugate(1) = sill_prior(1,0) + as_scalar(sills_current(n_gev)*quadraticForm/2);
          sills_current(n_gev) = 1/R::rgamma(sill_conjugate(0), 1/sill_conjugate(1));
          
          // New covariance matrix
          if (latent_processes_correlation_type==corr_expo) {GEVcovarMatrices.slice(n_gev) = sills_current(n_gev) * exp(-dss/ranges_current(n_gev));}
          if (latent_processes_correlation_type==corr_gauss) {GEVcovarMatrices.slice(n_gev) = sills_current(n_gev) * exp(-pow(dss/ranges_current(n_gev),2.0)/2);}
          if (latent_processes_correlation_type==corr_mat32) {GEVcovarMatrices.slice(n_gev) = sills_current(n_gev) * (1+dss/ranges_current(n_gev)*sqrt(3.0)) % exp(-dss/ranges_current(n_gev)*sqrt(3.0));}
          if (latent_processes_correlation_type==corr_mat52) {GEVcovarMatrices.slice(n_gev) = sills_current(n_gev) * (1+dss/ranges_current(n_gev)*sqrt(5.0) + 5.0/3.0*pow(dss/ranges_current(n_gev), 2.0)) % exp(-dss/ranges_current(n_gev)*sqrt(5.0));}
          
          
          
          
          // ---------- STEP 4: Updating the range
          // Candidates
          range_candidate = exp(R::rnorm(log(ranges_current(n_gev)), range_random_walk(n_gev)));
          if (latent_processes_correlation_type==corr_expo) {covarMatrix_candidate = sills_current(n_gev) * exp(-dss/range_candidate);}
          if (latent_processes_correlation_type==corr_gauss) {covarMatrix_candidate = sills_current(n_gev) * exp(-pow(dss/range_candidate, 2.0)/2.0);}
          if (latent_processes_correlation_type==corr_mat32) {covarMatrix_candidate = sills_current(n_gev) * (1+dss/range_candidate*sqrt(3.0)) % exp(-dss/range_candidate*sqrt(3.0));}
          if (latent_processes_correlation_type==corr_mat52) {covarMatrix_candidate = sills_current(n_gev) * (1+dss/range_candidate*sqrt(5.0) + 5.0/3.0*pow(dss/range_candidate, 2.0)) % exp(-dss/range_candidate*sqrt(5.0));}
          
          
          // Log-ratio
          range_logratio = as_scalar(-0.5*((GEV_current.col(n_gev)-GEVmeanVectors.col(n_gev)).t() *
            (inv(covarMatrix_candidate)-inv(GEVcovarMatrices.slice(n_gev))) *
            (GEV_current.col(n_gev)-GEVmeanVectors.col(n_gev))))
            -0.5*log(det(covarMatrix_candidate)) + 0.5*log(det(GEVcovarMatrices.slice(n_gev)))
            + R::dbeta(range_candidate/(2*distMax), range_prior(0,n_gev), range_prior(1,n_gev), 1)
            - R::dbeta(ranges_current(n_gev)/(2*distMax), range_prior(0,n_gev), range_prior(1,n_gev), 1)
            + log(range_candidate) - log(ranges_current(n_gev))
            ;
          
          // Acceptance test
          if ( (exp(range_logratio)>R::runif(0,1)) & (arma::is_finite((range_logratio))) ) {
            ranges_current(n_gev) = range_candidate;
            GEVcovarMatrices.slice(n_gev) = covarMatrix_candidate;
          }
        }
        
        
        // -------------------- IF the GEV parameter is CONSTANT (spatially) --------------------
        if (gev_vary(n_gev)==0) {
          GEV_candidate = GEV_current;
          GEV_candidate.col(n_gev) = GEV_current.col(n_gev) + R::rnorm(0, gev_random_walk(n_gev));
          llik_candidate = logLikelihood_cpp(Y, GEV_candidate, alpha_current, theta_current, nas); llik_count++; // Optimisable selon le paramètre regardé ?
          
          GEV_logratio = llik_candidate-llik_current
            + R::dnorm(GEV_candidate(0,n_gev), constant_gev_prior(0,n_gev), constant_gev_prior(1,n_gev), 1)
            - R::dnorm(GEV_current(0,n_gev), constant_gev_prior(0,n_gev), constant_gev_prior(1,n_gev), 1)
            ;
          
          if ( (exp(GEV_logratio)>R::runif(0,1)) & (arma::is_finite(GEV_logratio)) ) {
            GEV_current.col(n_gev) = GEV_candidate.col(n_gev);
            llik_current = llik_candidate;
          }
          
        }
        
        
      }
      
      // GEV timer
      GEV_timer(R) = timer.toc();
      
      
    } // -------------- END OF THINNING
    
    if (R>=nburn) {
      llik_chain(R-nburn) = llik_current;
      GEV_chain.slice(R-nburn) = GEV_current;
      alpha_chain(R-nburn) = alpha_current;
      tau_chain(R-nburn) = tau_current;
      sills_chain.row(R-nburn) = sills_current.t();
      ranges_chain.row(R-nburn) = ranges_current.t();
      A_chain.slice(R-nburn) = A_current;
      
      BETA_chain.slice(0).row(R-nburn) = BETA_current.col(0).t();
      BETA_chain.slice(1).row(R-nburn) = BETA_current.col(1).t();
      BETA_chain.slice(2).row(R-nburn) = BETA_current.col(2).t();
    }
    
    if( ((R+1)%trace==0) & !quiet) {Rcout << "Iter " << R+1 << ": " << llik_current << std::endl;}
    if( (R==(niter-1)) & ((R+1)%trace!=0) & !quiet) {Rcout << "Iter " << R+1 << ": " << llik_current << std::endl;}
  }
  
  
  
  // Information about time and acceptance rates
  double total_time = sum(A_timer+B_timer+alpha_timer+tau_timer+GEV_timer);
  if (!quiet) {
    // Time information
    Rcout << std::endl << "Time elapsed :" << std::endl;
    Rcout << "A     : " << sum(A_timer) << " sec (" << sum(A_timer)/total_time*100 << " %)" << std::endl;
    Rcout << "B     : " << sum(B_timer) << " sec (" << sum(B_timer)/total_time*100 << " %)" << std::endl;
    Rcout << "ALPHA : " << sum(alpha_timer) << " sec (" << sum(alpha_timer)/total_time*100 << " %)" << std::endl;
    Rcout << "TAU   : " << sum(tau_timer) << " sec (" << sum(tau_timer)/total_time*100 << " %)" << std::endl;
    Rcout << "GEV   : " << sum(GEV_timer) << " sec (" << sum(GEV_timer)/total_time*100 << " %)" << std::endl;
    Rcout << "----------------" << std::endl;
    Rcout << "TOTAL : " << total_time << std::endl;
    
    // Likelihood computations
    Rcout << std::endl << "Likelihood computed " << llik_count << " times" << std::endl;
    Rcout << "----------------" << std::endl;
  }
  
  
  
  /* ***************************
   *****  END OF FUNCTION  *****
   ************************** */
  
  // Spatial parameters:
  Rcpp::List spatial_parameters_list = Rcpp::List::create(gev_vary, BETA_chain, sills_chain, ranges_chain);
  CharacterVector spatial_parameters_list_names(4);
  spatial_parameters_list_names(0) = "vary";
  spatial_parameters_list_names(1) = "beta";
  spatial_parameters_list_names(2) = "sills";
  spatial_parameters_list_names(3) = "ranges";
  spatial_parameters_list.attr("names") = spatial_parameters_list_names;
  
  
  // Result:
  Rcpp::List result = Rcpp::List::create(GEV_chain, alpha_chain, tau_chain, A_chain, llik_chain, total_time, spatial_parameters_list);
  CharacterVector mainResult_names(7);
  mainResult_names(0) = "GEV";
  mainResult_names(1) = "alpha";
  mainResult_names(2) = "tau";
  mainResult_names(3) = "A";
  mainResult_names(4) = "llik";
  mainResult_names(5) = "time";
  mainResult_names(6) = "spatial";
  result.attr("names") = mainResult_names;
  
  return result;
}

