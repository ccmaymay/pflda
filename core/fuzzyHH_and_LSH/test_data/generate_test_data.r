# Keith Levin
# klevin@jhu.edu
# Johns Hopkins University, CLSP, HLTCOE
# September, 2013

# Code for generating a large number of draws from a mixture of
# many Gaussians.
library(MASS)

get.sample <- function()
{
  # seed the generator for debugging.
  set.seed(1969); # in the sunshine

  # we have k cluster centers in d-dimensional space.
  k = 30;
  d = 2;
  # and we draw n times from a mixture of Gaussians over these.
  n = 10000;
  # each cluster center is the mean of a spherical Gaussian.
  componentSigma = 0.01;
  # the centers are themselves draw from a hihg-variance gaussian.
  overallSigma = 10.0;

  # generate probabilities of drawing from each of the centers.
  probs <- 0.9^(1:k);
  probs <- probs/sum(probs);

  DATA <- generate.data(n, rep(0,d), overallSigma, probs, componentSigma);
}

# This function adapted from code at
# http://www.cs.princeton.edu/courses/archive/spring12/cos424/r/mixt-gaussians-demo.R
generate.data <- function(n, prior.mean, prior.var, priors, comp.var)
# n : number of draws
# prior.mean : mean of distribution from which to generate centers
# prior.var : variance of said distribution.
# priors : prob. vector determining prior center probabilities.
#	priors[i] is the i-th mixing coefficient.
# comp.var : variance of each individual Gaussian (i.e., all components
#	generate from a different cluster center).
{
  if( length(prior.var) != 1 & length(prior.mean) != length(prior.var) )
  { stop() }
  k <- length(priors) # number of centers.
  p <- length(prior.mean) # dimensionality of space we're in.
  # generate mixture locations from the prior
  locs <- mvrnorm(k, mu=prior.mean, Sigma=diag(prior.var, p))
  #vars <- rgamma(k, shape=1, scale=0.25)
  vars <- rep(comp.var,k)
  # generate the data
  obs <- matrix(0, nrow=n, ncol=p)
  z <- numeric(n)
  for (i in 1:n)
  {
    # draw the cluster uniformly at random
    #z[i] <- sample(1:k, 1)
    # draw according to specified prior distribution.
    z[i] <- which(rmultinom(1, 1, priors) != 0)
    # draw the observation from the corresponding mixture location
    obs[i,] <- mvrnorm(1, mu=locs[z[i],], Sigma=diag(vars[z[i]],p))
  }
  list(locs=locs, vars=vars, z=z, obs=obs, priors=priors)
}
