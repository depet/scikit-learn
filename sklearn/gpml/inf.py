import numpy

from . import cov
from . import lik
from . import mean
from . import util

# Exact inference for a GP with Gaussian likelihood. Compute a parametrization
# of the posterior, the negative log marginal likelihood and its derivatives
# w.r.t. the hyperparameters.
def exact(hyp, meanf, covf, likf, x, y, nargout=None):
  if isinstance(likf, tuple) and likf[0] != lik.gauss or not isinstance(likf, tuple) and likf != lik.gauss:
    raise AttributeError('Exact inference only possible with Gaussian likelihood')

  if nargout is None:
    nargout = 3
  
  n, D = numpy.shape(x)
  
  K = cov.feval(covf, hyp['cov'], x)  
  m = mean.feval(meanf, hyp['mean'], x)
  
  sn2 = numpy.exp(2*hyp['lik'])
  L = numpy.linalg.cholesky(K/sn2+numpy.eye(n)).T
  alpha = util.solve_chol(L,y-m)/sn2
  
  sW = numpy.ones((n,1))/numpy.sqrt(sn2)
  
  post = {'alpha': alpha, 'sW': sW, 'L': L}
  
  r = post
  
  if nargout > 1:
    nlZ = numpy.dot((y-m).T, alpha/2) + numpy.sum(numpy.log(numpy.array([numpy.diag(L)]).T)) + n*numpy.log(2*numpy.pi*sn2)/2
    r = (post, nlZ)
  if nargout > 2:
    dnlZ = {'cov': numpy.zeros(numpy.shape(hyp['cov'])), 'lik': numpy.zeros(numpy.shape(hyp['lik'])), 'mean': numpy.zeros(numpy.shape(hyp['mean']))}
    Q = util.solve_chol(L, numpy.eye(n))/sn2 - numpy.dot(alpha,alpha.T)
    for i in range(numpy.size(hyp['cov'])):
      dnlZ['cov'][i] = numpy.sum(numpy.sum(Q*cov.feval(covf, hyp['cov'], x, None, i)))/2
    dnlZ['lik'] = sn2*numpy.trace(Q)
    for i in range(numpy.size(hyp['mean'])):
      dnlZ['mean'][i] = numpy.dot(-mean.feval(meanf, hyp['mean'], x, None, i).T, alpha)
    r = r + (dnlZ,)

  return r
