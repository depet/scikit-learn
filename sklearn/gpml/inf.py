import numpy

from . import cov
from . import lik
from . import mean
from . import util

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
    nlZ = numpy.dot((y-m).T, alpha/2) + numpy.sum(numpy.log(numpy.array([numpy.diag(L)]).T)) + n*numpy.log(2*numpy.pi*sn2)/2
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


