"""
Inference methods compute the (approximate) posterior for a Gaussian process.
Methods currently implemented include::

simple mean functions::

  exact       - Exact inference (only possible with Gaussian likelihood)
  laplace     - Laplace's Approximation
  EP          - Expectation Propagation
  VB          - Variational Bayes Approximation
  
  fitc        - Large scale inference with approximate covariance matrix
  fitcLaplace - Large scale inference with approximate covariance matrix
  fitcEP      - Large scale inference with approximate covariance matrix

  mcmc        - Markov Chain Monte Carlo and Annealed Importance Sampling
                No derivatives w.r.t. to hyperparameters are provided
  
  loo         - Leave-One-Out predictive probability and
                Least-Squares Approximation
"""
# Author: Dejan Petelin <http://www.linkedin.com/in/dejanpetelin>
# Python implementation of the GPML MATLAB Toolbox, version 3.2
# License: see copyright

import numpy

from . import cov
from . import lik
from . import mean
from . import util

__ep_last_ttau = numpy.array([[]])
__ep_last_tnu = numpy.array([[]])

def exact(hyp, meanf, covf, likf, x, y, nargout=None):
  """
  Exact inference for a GP with Gaussian likelihood. Compute a parametrization
  of the posterior, the negative log marginal likelihood and its derivatives
  w.r.t. the hyperparameters.
  """
  if isinstance(likf, tuple) and likf[0] != lik.gauss or not isinstance(likf, tuple) and likf != lik.gauss:
    raise AttributeError('Exact inference only possible with Gaussian likelihood')

  if nargout is None:
    nargout = 3
  
  n, D = numpy.shape(x)
  
  # evaluate covariance matrix
  K = cov.feval(covf, hyp['cov'], x)
  # evaluate mean vector
  m = mean.feval(meanf, hyp['mean'], x)
  
  # extract likelihood hyperparameter value and solve cholesky factorization
  sn2 = numpy.exp(2*hyp['lik'])
  L = numpy.linalg.cholesky(K/sn2+numpy.eye(n)).T
  alpha = util.solve_chol(L,y-m)/sn2
  
  sW = numpy.ones((n,1))/numpy.sqrt(sn2)
  
  # construct dict containing posterior parameters
  post = {'alpha': alpha, 'sW': sW, 'L': L}
  
  r = post
  
  if nargout > 1:
    # negative log marginal likelihood
    nlZ = float(numpy.dot((y-m).T, alpha/2) + numpy.sum(numpy.log(numpy.array([numpy.diag(L)]).T)) + n*numpy.log(2*numpy.pi*sn2)/2)
    r = (post, nlZ)
  if nargout > 2:
    # derivatives
    dnlZ = {'cov': numpy.zeros(numpy.shape(hyp['cov'])), 'lik': numpy.zeros(numpy.shape(hyp['lik'])), 'mean': numpy.zeros(numpy.shape(hyp['mean']))}
    Q = util.solve_chol(L, numpy.eye(n))/sn2 - numpy.dot(alpha,alpha.T)
    for i in range(numpy.size(hyp['cov'])):
      dnlZ['cov'][i] = numpy.sum(numpy.sum(Q*cov.feval(covf, hyp['cov'], x, None, i)))/2
    dnlZ['lik'] = sn2*numpy.trace(Q)
    for i in range(numpy.size(hyp['mean'])):
      dnlZ['mean'][i] = numpy.dot(-mean.feval(meanf, hyp['mean'], x, None, i).T, alpha)
    r = r + (dnlZ,)

  return r


def ep(hyp, meanf, covf, likf, x, y, nargout=2):
  """
  Expectation Propagation approximation to the posterior Gaussian Process.
  The function takes a specified covariance function and likelihood 
  function. In the EP algorithm, the sites are updated in random order, 
  for better performance when cases are ordered according to the targets.
  """
  global __ep_last_ttau
  global __ep_last_tnu

  tol = 1e-4
  max_sweep = 10
  min_sweep = 2

  inff = 'ep'
  n = numpy.size(x,0)
  K = cov.feval(covf, hyp['cov'], x)
  m = mean.feval(meanf, hyp['mean'], x)

  # A note on naming: variables are given short but descriptive names in 
  # accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
  # and s2 are mean and variance, nu and tau are natural parameters. A leading t
  # means tilde, a subscript _ni means "not i" (for cavity parameters), or _n
  # for a vector of cavity parameters.
  
  # marginal likelihood for ttau = tnu = zeros((n,1))
  nlZ0 = float(-numpy.sum(lik.feval(likf, hyp['lik'], y, m, numpy.reshape(numpy.diag(K),(-1,1)), 'ep', nargout=1)))
  if numpy.shape(__ep_last_ttau) != (n, 1):
    ttau = numpy.zeros((n,1))
    tnu = numpy.zeros((n,1))
    Sigma = K.copy()
    mu = numpy.zeros((n,1))
    nlZ = nlZ0
  else:
    ttau = __ep_last_ttau.copy()
    tnu = __ep_last_tnu.copy()
    Sigma, mu, nlZ, L = __epComputeParams(K, y, ttau, tnu, likf, hyp, m, inff)
    # if zero is better, initialize with zero instead
    if nlZ > nlZ0:
      ttau = numpy.zeros((n,1))
      tnu = numpy.zeros((n,1))
      Sigma = K.copy()
      mu = numpy.zeros((n,1))
      nlZ = nlZ0

  nlZ_old = numpy.Inf
  sweep = 0
  # converged, max. sweeps or min. sweeps?
  while (numpy.abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or sweep < min_sweep:
    nlZ_old = nlZ
    sweep += 1
    # iterate EP updates (in random order) over examples
    for i in numpy.random.permutation(n):
      # first find the cavity distribution, params tau_ni and nu_ni
      tau_ni = numpy.reshape(1/Sigma[i,i] - ttau[i],(-1,1))
      nu_ni = numpy.reshape(mu[i]/Sigma[i,i] + m[i]*tau_ni - tnu[i],(-1,1))

      # compute the desired derivatives of the indivdual log partition function
      lZ, dlZ, d2lZ = lik.feval(likf, hyp['lik'], y[i], nu_ni/tau_ni, 1/tau_ni, inff, nargout=3)
      ttau_old = ttau[i].copy()
      
      # enforce positivity i.e. lower bound ttau by zero
      ttau[i] = numpy.max(-d2lZ/(1+d2lZ/tau_ni),0)
      tnu[i] = (dlZ + (m[i]-nu_ni/tau_ni)*d2lZ )/(1+d2lZ/tau_ni)

      # finally rank-1 update Sigma
      ds2 = ttau[i] - ttau_old
      si = numpy.reshape(Sigma[:,i],(-1,1))
      # takes 70% of total time
      Sigma = Sigma - ds2/(1+ds2*si[i])*numpy.dot(si,si.T)
      mu = numpy.dot(Sigma,tnu)
    Sigma, mu, nlZ, L = __epComputeParams(K, y, ttau, tnu, likf, hyp, m, inff)

  if sweep == max_sweep:
    raise Error('Maximum number of sweeps reached in EP inference function.')

  __ep_last_ttau = ttau.copy()
  __ep_last_tnu = tnu.copy()

  # posterior parameters
  sW = numpy.sqrt(ttau)
  alpha = tnu-sW*util.solve_chol(L,sW*numpy.dot(K,tnu))
  post = {}
  post['sW'] = sW
  post['alpha'] = alpha
  post['L'] = L

  res = post
  if nargout > 1:
    res = (post, nlZ)

  # do we want derivatives?
  if nargout > 2:
    dnlZ = {'cov': numpy.zeros(numpy.shape(hyp['cov'])), 'lik': numpy.zeros(numpy.shape(hyp['lik'])), 'mean': numpy.zeros(numpy.shape(hyp['mean']))}
    V = numpy.linalg.solve(L.T,numpy.tile(sW,(1,n))*K)
    Sigma = K - numpy.dot(V.T,V)
    mu = numpy.dot(Sigma,tnu)

    # compute the log marginal likelihood and vectors of cavity parameters
    tau_n = 1/numpy.reshape(numpy.diag(Sigma),(-1,1))-ttau
    nu_n = mu/numpy.reshape(numpy.diag(Sigma),(-1,1))-tnu+m*tau_n

    # covariance hyperparameters
    F = numpy.dot(alpha,alpha.T) - numpy.tile(sW,(1,n)) * util.solve_chol(L,numpy.diagflat(sW))
    for i in range(numpy.size(hyp['cov'])):
      dK = cov.feval(covf, hyp['cov'], x, None, i)
      dnlZ['cov'][i] = -numpy.sum(F*dK)/2
    # likelihood hyperparameters
    for i in range(numpy.size(hyp['lik'])):
      dlik = lik.feval(likf, hyp['lik'], y, nu_n/tau_n, 1/tau_n, inff, i)
      dnlZ['lik'][i] = -numpy.sum(dlik)
    # mean hyperparameters
    junk, dlZ = lik.feval(likf, hyp['lik'], y, nu_n/tau_n, 1/tau_n, inff, nargout=2)
    for i in range(numpy.size(hyp['mean'])):
      dm = mean.feval(meanf, hyp['mean'], x, i)
      dnlZ['mean'][i] = numpy.dot(-dlZ.T,dm)
    res += (dnlZ,)

  return res


def __epComputeParams(K, y, ttau, tnu, likf, hyp, m, inff):
  """
  Function to compute the parameters of the Gaussian approximation, Sigma and
  mu, and the negative log marginal likelihood, nlZ, from the current site
  parameters, ttau and tnu. Also returns L (useful for predictions).
  """
  n = numpy.size(y)
  sW = numpy.sqrt(ttau)
  # L'*L=B=eye(n)+sW*K*sW
  L = numpy.linalg.cholesky(numpy.eye(n)+numpy.dot(sW,sW.T)*K).T
  V = numpy.linalg.solve(L.T,(numpy.tile(sW,(1,n))*K))
  Sigma = K - numpy.dot(V.T,V)
  mu = numpy.dot(Sigma,tnu)

  # compute the log marginal likelihood
  tau_n = 1/numpy.reshape(numpy.diag(Sigma),(-1,1))-ttau
  # vectors of cavity parameters
  nu_n  = mu/numpy.reshape(numpy.diag(Sigma),(-1,1))-tnu+m*tau_n
  lZ = lik.feval(likf, hyp['lik'], y, nu_n/tau_n, 1/tau_n, inff, nargout=1)
  nlZ = float(numpy.sum(numpy.log(numpy.diag(L))) - numpy.sum(lZ) - numpy.dot(tnu.T,numpy.dot(Sigma,tnu))/2 - numpy.dot((nu_n-m*tau_n).T, (ttau/tau_n*(nu_n-m*tau_n)-2*tnu) / (ttau+tau_n))/2 + numpy.sum(numpy.power(tnu,2)/(tau_n+ttau))/2 - numpy.sum(numpy.log(1+ttau/tau_n))/2)
  return (Sigma, mu, nlZ, L)


def fitc(hyp, meanf, covf, likf, x, y, nargout=None):
  """
  FITC approximation to the posterior Gaussian process. The function is
  equivalent to inf.exact with the covariance function:
    Kt = Q + G,
    G = diag(g),
    g = diag(K-Q),
    Q = Ku'*inv(Quu)*Ku,
  where Ku and Kuu are covariances w.r.t. to inducing inputs xu, snu2 = sn2/1e6
  is the noise of the inducing inputs and Quu = Kuu + snu2*eye(nu).
  
  We fixed the standard deviation of the inducing inputs snu to be a one per mil
  of the measurement noise's standard deviation sn.
  
  The implementation exploits the Woodbury matrix identity
    inv(Kt) = inv(G) - inv(G)*V'*inv(eye(nu)+V*inv(G)*V')*V*inv(G)
  in order to be applicable to large datasets. The computational complexity
  is O(n nu^2) where n is the number of data points x and nu the number of
  inducing inputs in xu.
  
  The function takes a specified covariance function and likelihood function, 
  and is designed to be used with GP and in conjunction with cov.fitc and 
  lik.gauss.
  """
  if isinstance(likf, tuple) and likf[0] != lik.gauss or not isinstance(likf, tuple) and likf != lik.gauss:
    raise AttributeError('Inference with FITC only possible with Gaussian likelihood.')

  if isinstance(covf, tuple) and covf[0] != cov.fitc or not isinstance(covf, tuple) and covf != cov.fitc:
    raise AttributeError('Only cov.fitc supported.')

  if nargout is None:
    nargout = 3

  # evaluate covariance matrix
  diagK, Kuu, Ku = cov.feval(covf, hyp['cov'], x, nargout=3)
  # evaluate mean vector
  m = mean.feval(meanf, hyp['mean'], x)
  
  n, D = numpy.shape(x)
  nu = numpy.size(Kuu,0)

  sn2  = numpy.exp(2*hyp['lik'])
  snu2 = 1e-6*sn2
  Luu  = numpy.linalg.cholesky(Kuu+snu2*numpy.eye(nu)).T
  V  = numpy.linalg.solve(Luu.T,Ku)
  g_sn2 = diagK + sn2 - numpy.reshape(numpy.sum(V*V,0),(-1,1))
  Lu = numpy.linalg.cholesky(numpy.eye(nu) + numpy.dot(V/numpy.tile(g_sn2.T,(nu,1)),V.T)).T
  r = (y-m)/numpy.sqrt(g_sn2)
  be = numpy.linalg.solve(Lu.T,numpy.dot(V,(r/numpy.sqrt(g_sn2))))
  iKuu = util.solve_chol(Luu,numpy.eye(nu))

  # construct dict containing posterior parameters
  alpha = numpy.linalg.solve(Luu,numpy.linalg.solve(Lu,be))
  L = util.solve_chol(numpy.dot(Lu,Luu),numpy.eye(nu)) - iKuu
  sW = numpy.ones((n,1))/numpy.sqrt(sn2) # unused for FITC prediction
  post = {'alpha': alpha, 'sW': sW, 'L': L}

  res = post

  if nargout > 1:
    # negative log marginal likelihood
    nlZ = numpy.sum(numpy.log(numpy.diag(Lu))) + (numpy.sum(numpy.log(g_sn2)) + n*numpy.log(2*numpy.pi) + numpy.dot(r.T,r) - numpy.dot(be.T,be))/2
    nlZ = nlZ[0,0]
    res = (post, nlZ)
    if nargout > 2:
      dnlZ = {'cov': numpy.zeros(numpy.shape(hyp['cov'])), 'lik': numpy.zeros(numpy.shape(hyp['lik'])), 'mean': numpy.zeros(numpy.shape(hyp['mean']))}
      al = r/numpy.sqrt(g_sn2) - numpy.dot(V.T,(numpy.linalg.solve(Lu,be)))/g_sn2
      B = numpy.dot(iKuu,Ku)
      w = numpy.dot(B,al)
      W = numpy.linalg.solve(Lu.T,(V/numpy.tile(g_sn2.T,(nu,1))))
      # covariance derivatives
      for i in range(numpy.size(hyp['cov'])):
        ddiagKi, dKuui, dKui = cov.feval(covf, hyp['cov'], x, None, i, nargout=3)
        R = 2*dKui-numpy.dot(dKuui,B)
        v = ddiagKi - numpy.reshape(numpy.sum(R*B,0),(-1,1))
        dnlZ['cov'][i] = float(numpy.dot(ddiagKi.T,1/g_sn2) + numpy.dot(w.T,numpy.dot(dKuui,w)-2*numpy.dot(dKui,al))-numpy.dot(al.T,v*al) - numpy.dot(numpy.sum(W*W,0),v) - numpy.sum(numpy.dot(R,W.T)*numpy.dot(B,W.T)))/2
      # likelihood derivative
      dnlZ['lik'][0] = sn2*float(numpy.sum(1/g_sn2) - numpy.sum(numpy.sum(W*W,0)) - numpy.dot(al.T,al))
      # since snu2 is a fixed fraction of sn2, there is a covariance-like term in the derivative as well
      dKuui = 2*snu2
      R = -dKuui*B
      v = numpy.reshape(-numpy.sum(R*B,0),(-1,1))
      dnlZ['lik'][0] = dnlZ['lik'][0] + (numpy.dot(w.T*dKuui,w) - numpy.dot(al.T,v*al) - numpy.dot(numpy.sum(W*W,0),v) - numpy.sum(numpy.dot(R,W.T)*numpy.dot(B,W.T)))/2
      # mean derivatives
      for i in range(numpy.size(hyp['mean'])):
        dnlZ['mean'][i] = numpy.dot(-mean.feval(meanf, hyp['mean'], x, i).T,al)
      res = (post, nlZ, dnlZ)

  return res


def mcmc(hyp, meanf, covf, likf, x, y, par=None, nargout=None):
  """
  Markov Chain Monte Carlo (MCMC) sampling from posterior and 
  Annealed Importance Sampling (AIS) for marginal likelihood estimation.

  The algorithms are not to be used as a black box, since the acceptance rate
  of the samplers need to be carefully monitored. Also, there are no derivatives 
  of the marginal likelihood available.

  There are additional parameters:
    - par['sampler']   switch between the samplers 'hmc' or 'ess'
    - par['Nsample']   num of samples
    - par['Nskip']     num of steps out of which one sample kept
    - par['Nburnin']   num of burn in samples (corresponds to Nskip*Nburning steps)
    - par['Nais']      num of AIS runs to remove finite temperature bias
  Default values are sampler=hmc, Nsample=200, Nskip=40, Nburnin=10, Nais=3.

  The Hybrid Monte Carlo Sampler (HMC) is implemented as described in the
  technical report: Probabilistic Inference using MCMC Methods by Radford Neal,
  CRG-TR-93-1, 1993.

      Instead of sampling from f ~ 1/Zf * N(f|m,K) P(y|f), we use a 
      parametrisation in terms of alpha = inv(K)*(f-m) and sample from 
      alpha ~ P(a) = 1/Za * N(a|0,inv(K)) P(y|K*a+m) to increase numerical 
      stability since log P(a) = -(a'*K*a)/2 + log P(y|K*a+m) + C and its 
      gradient can be computed safely.

      The leapfrog stepsize as a time discretisation stepsize comes with a 
      tradeoff:
        - too small: frequently accept, slow exploration, accurate dynamics
        - too large: seldomly reject, fast exploration, inaccurate dynamics
      In order to balance between the two extremes, we adaptively adjust the
      stepsize in order to keep the acceptance rate close to a target acceptance
      rate. Taken from http://deeplearning.net/tutorial/hmc.html.

      The code issues a warning in case the overall acceptance rate did deviate
      strongly from the target. This can indicate problems with the sampler.

  The Elliptical Slice Sampler (ESS) is a straight implementation that is
  inspired by the code shipped with the paper: Elliptical slice sampling by
  Iain Murray, Ryan Prescott Adams and David J.C. MacKay, AISTATS 2010.

  Annealed Importance Sampling (AIS) to determine the marginal likelihood is
  described in the technical report: AIS, Radford Neal, 1998.
  """
  if par is None:
    par = {}
  if 'sampler' in par:
    alg = par['sampler']
  else:
    alg = 'hmc'
  if 'Nsample' in par:
    N = par['Nsample']
  else:
    N = 200
  if 'Nskip' in par:
    Ns = par['Nskip']
  else:
    Ns = 40
  if 'Nburnin' in par:
    Nb =par['Nburnin']
  else:
    Nb = 10
  if 'Nais' in par:
    R = par['Nais']
  else:
    R = 3

  # evaluate the covariance matrix
  K = cov.feval(covf, hyp['cov'], x)
  # evaluate the mean vector
  m = mean.feval(meanf, hyp['mean'], x)
  n = numpy.size(K,0)
  # try an ordinary Cholesky decomposition
  try:
    cK = numpy.linalg.cholesky(K).T
  except numpy.linalg.LinAlgError:
    # regularise
    sr2 = 1e-8*numpy.sum(numpy.diag(K))/n
    cK = numpy.linalg.cholesky(K+sr2*numpy.eye(n)).T
  # overall number of steps
  T = numpy.dot(N+Nb,Ns)
  # sample w/o annealing
  alpha, Na = __sample(K,cK,m,y,likf,hyp['lik'],N,Nb,Ns,alg)
  post = {}
  post['alpha'] = alpha
  al = numpy.reshape(numpy.sum(alpha,1)/float(N),(-1,1))
  # inv(K) - cov(alpha)
  post['L'] = -(numpy.linalg.solve(cK,numpy.linalg.solve(cK.T,numpy.eye(n))) + numpy.dot(al,al.T) - numpy.dot(alpha,alpha.T)/float(N))
  post['sW'] = numpy.array([[]])
  # additional output parameter
  post['acceptance_rate_MCMC'] = float(Na)/T

  res = post

  # annealed importance sampling
  if nargout > 1:
    # discrete time t from 1 to T and temperature tau from tau(1)=0 to tau(T)=1
    # annealing schedule, effort at start
    taus = numpy.power(numpy.concatenate((numpy.zeros((1,Nb)), numpy.linspace(0,1,N).reshape((1,-1))), 1),4)
    # the fourth power idea is taken from the Kuss&Rasmussen paper, 2005 JMLR
    # Z(t) := \int N(f|m,K) lik(f)^taus(t) df hence Z(1) = 1 and Z(T) = Z
    # Z = Z(T)/Z(1) = prod_t Z(t)/Z(t-1); 
    # ln Z(t)/Z(t-1) = ( tau(t)-tau(t-1) ) * loglik(f_t)
    lZ = numpy.zeros((R,1))
    # we have: sum(dtaus)==1
    dtaus = numpy.diff(taus[:,Nb+numpy.arange(N)])
    post['acceptance_rate_AIS'] = numpy.zeros((3,1))
    for r in range(R):
      # AIS
      A, Na = __sample(K,cK,m,y,likf,hyp['lik'],N,Nb,Ns,alg,taus)
      # evaluate the likelihood sample-wise      
      for t in range(1,N):
        lp = lik.feval(likf,hyp['lik'],y,numpy.dot(K,A[:,[t]])+m,numpy.array([[]]),'laplace',nargout=1)
        lZ[r,0] = lZ[r,0] + dtaus[0,t-1]*numpy.sum(lp)
      # additional output parameters
      post['acceptance_rate_AIS'][r,0] = float(Na)/T
    # remove finite temperature bias, softmax average    
    nlZ = numpy.log(R) - __logsumexp(lZ)
    res = (post, nlZ)
    # marginal likelihood derivatives are not computed
    if nargout > 2:
      dnlZ = {'cov': numpy.zeros(numpy.shape(hyp['cov'])), 'lik': numpy.zeros(numpy.shape(hyp['lik'])), 'mean': numpy.zeros(numpy.shape(hyp['mean']))}
      res += (dnlZ,)
  return res


def __sample(K, cK, m, y, likf, hyp, N, Nb, Ns, alg, taus=None):
  """
  Choose between HMC and ESS depending on the alg string.
  """
  if alg == 'hmc':
    return __sample_hmc(K, m, y, likf, hyp, N, Nb, Ns, taus)
  else:
    return __sample_ess(cK, m, y, lik, hyp, N, Nb, Ns, taus)

def __sample_hmc(K, m, y, likf, hyp, N, Nb, Ns, taus=None):
  """
  Sample using Hamiltonian dynamics as proposal algorithm.
  """  
  # use adaptive stepsize rule to enforce a specific acceptance rate 
  epmin = 1e-6                                      # minimum leapfrog stepsize
  epmax = 9e-1                                      # maximum leapfrog stepsize
  ep = 1e-2                                         # initial leapfrog stepsize
  acc_t = 0.9                                          # target acceptance rate
  acc = 0                                             # current acceptance rate
  epinc = 1.02 # increase factor of stepsize if acceptance rate is below target
  epdec = 0.98 # decrease factor of stepsize if acceptance rate is above target
  lam = 0.01    # exponential moving average computation of the acceptance rate
                                                 # 2/(3*lam) steps half height;
                                      # lam/nstep = 0.02/33, 0.01/66, 0.005/133
  l = 20                     # number of leapfrog steps to perform for one step
  n = numpy.size(K,0)
  T = (N+Nb)*Ns                                       # overall number of steps
  alpha = numpy.zeros((n,N))                                    # sample points
  al = numpy.zeros((n,1))                                    # current position
  # default is no annealing
  if taus is not None:
    tau = taus[0,0]
  else:
    tau = 1
  gold, eold = __E(al, K, m, y, likf, hyp, tau, 2)   # initial energy, gradient  
  Na = 0                                            # number of accepted points
  for t in range(T):
    # parameter from schedule
    if taus is not None:
      tau = taus[0,numpy.floor(t/Ns)]
    p = numpy.random.randn(n,1)                       # random initial momentum
    q = al
    g = gold
    Hold = numpy.dot(p.T,p)/2. + eold          # H = Ekin + Epot => Hamiltonian
    # leapfrog discretization steps, Euler like
    for ll in range(l):
      p = p - (ep/2.)*g                               # half step in momentum p
      q = q + ep*p                                    # full step in position q
      g = __E(q, K, m, y, likf, hyp, tau)             # compute new gradient  g
      p = p - (ep/2.)*g                               # half step in momentum p
    g, e = __E(q, K, m, y, likf, hyp, tau, 2)
    H = numpy.dot(p.T,p)/2. + e                         # recompute Hamiltonian
    acc = (1-lam)*acc                           # decay current acceptance rate
    # accept with p = min(1,exp(Hold-H))
    if numpy.log(numpy.random.rand()) < Hold-H:
      al = q                                                  # keep new state,
      gold = g                                                       # gradient
      eold = e                                           # and potential energy
      acc = acc + lam                         # increase rate due to acceptance
      Na = Na + 1                              # increase number accepted steps
    # too large acceptance rate => increase leapfrog stepsize
    if acc > acc_t:
      ep = epinc*ep
    # too small acceptance rate => decrease leapfrog stepsize
    else:
      ep = epdec*ep
    # clip stepsize ep to [epmin,epmax]
    if ep < epmin:
      ep = epmin
    if ep > epmax:
      ep = epmax
    # keep one state out of Ns
    if numpy.mod(t+1,Ns) == 0:
      # wait for Nb burn-in steps
      if (t+1)/Ns > Nb:
        alpha[:,[(t+1)/Ns-Nb-1]] = al

  # Acceptance rate in the right ballpark?
  if float(Na)/T < acc_t*0.9 or 1.07*acc_t < float(Na)/T:
    print 'The acceptance rate %1.2f%% is not within' % (100*float(Na)/T,)
    print ' [%1.1f, %1.1f]%%\n' % (100*acc_t*0.9, 100*acc_t*1.07)
    if taus is None:
      print 'Bad (HMC) acceptance rate'
    else:
      print 'Bad (AIS) acceptance rate'

  return (alpha, Na)


def __E(al, K, m, y, likf, hyp, tau=None, nargout=1):
  """
  Potential energy function value and its gradient.
  """
  # E(f) = E(al) = al'*K*al/2 - tau*loglik(f), f = K*al+m
  # g(f) = g(al) = al - tau*dloglik(f) => dE/dal = K*( al-dloglik(f) )
  if tau is None:
    tau = 1
  if tau > 1:
    tau = 1
  if tau < 0:
    tau = 0
  Kal = numpy.dot(K,al)
  lp, dlp = lik.feval(likf, hyp, y, Kal+m, numpy.array([[]]), 'laplace', nargout=2)
  g = Kal - numpy.dot(K,numpy.dot(tau,dlp))
  res = g
  if nargout > 1:
    e = numpy.dot(al.T,Kal)/2. - tau*numpy.sum(lp)
    res = (g, e)
  return res


def __logsumexp(x):
  """
  y = logsumexp(x) = log(sum(exp(x(:))) avoiding overflow
  """
  xx = numpy.reshape(x.T,(-1,1))
  mx = numpy.max(xx)
  return numpy.log(numpy.sum(numpy.exp(xx-mx)))+mx


