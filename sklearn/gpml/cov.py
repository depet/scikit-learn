import numpy
import numpy.matlib

from . import util

# Squared Exponential covariance function with Automatic Relevance Detemination
# (ARD) distance measure. The covariance function is parameterized as:
#
# k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
#
# where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
# D is the dimension of the input space and sf2 is the signal variance. The
# hyperparameters are:
def seArd(hyp=None, x=None, z=None, hi=None, dg=None):
  # report number of parameters
  if x is None:
    return '(D+1)'
  
  if z is None:
    z = numpy.array([[]])
    
  if dg is None:
    dg = False
  
  xeqz = numpy.size(z) == 0

  n, D = numpy.shape(x)
  
  ell = numpy.exp(hyp[0:D])      # characteristic length scale
  sf2 = numpy.exp(2*hyp[D])        # signal variance

  # precompute squared distances
  if dg:
    K = numpy.zeros((numpy.size(x,0),1))
  else:
    if xeqz:
      K = util.sq_dist(numpy.dot(numpy.diagflat(1./ell),x.T))
    else:
      K = util.sq_dist(numpy.dot(numpy.diagflat(1./ell),x.T),numpy.dot(numpy.diagflat(1./ell),z.T))

  K = sf2*numpy.exp(-K/2)
  
  if hi is not None:
    if hi >= 0 and hi < D:
      if dg:
        K = K*0
      else:
        if xeqz:
          K = numpy.multiply(K, util.sq_dist(numpy.array([x[:,hi]])/ell[hi]))
        else:
          K = numpy.multiply(K, util.sq_dist(numpy.array([x[:,hi]])/ell[hi],numpy.array([z[:,hi]])/ell[hi]))
    elif hi == D:
      K = 2*K
    else:
      raise AttributeError('Unknown hyperparameter')

  return K



# Independent covariance function, ie "white noise", with specified variance.
# The covariance function is specified as:
#
# k(x^p,x^q) = s2 * \delta(p,q)
#
# where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
# which is 1 iff p=q and zero otherwise. Two data points p and q are considered
# equal if their norm is less than 1e-9. The hyperparameter is
#
# hyp = [ log(sqrt(s2)) ]
def noise(hyp=None, x=None, z=None, hi=None, dg=None):
  tol = 1e-9

  # report number of parameters
  if x is None:
    return '1'

  if z is None:
    z = numpy.array([[]])

  xeqz = numpy.size(z) == 0
  
  if x.ndim == z.ndim and numpy.shape(x) == numpy.shape(z):
    xeqz = numpy.linalg.norm(x.T-z.T, numpy.inf)
  
  n = numpy.size(x,0)
  s2 = numpy.exp(2*hyp)  # noise variance
  
  # precompute raw
  if dg:
    K = numpy.ones((n,1))
  else:
    if xeqz:
      K = numpy.eye(n)
    else:
      K = util.sq_dist(x.T,z.T) < tol*tol
      K = K.astype(float)
  
  if hi is None:
    K = s2*K
  else:
    if hi == 0:
      K = 2*s2*K
    else:
      raise AttributeError('Unknown hyperparameter')

  return K


# covSum - compose a covariance function as the sum of other covariance
# functions. This function doesn't actually compute very much on its own, it
# merely does some bookkeeping, and calls other covariance functions to do the
# actual work.
def sum(cov, hyp, x=None, z=None, hi=None):
  if cov is None:
    raise AttributeError('Covariance functions to be summed must be defined.')
  else:
    l = len(cov)
    if l == 0:
      raise AttributeError('At least one covariance function to be summed must be defined.')

  # iterate over covariance functions and collect number of hyperparameters
  for i in range(l):
    fun = cov(i)
    #j(i) = fun()

  # if there is no data, return number of hyperparameters
  if x is None:
    c = 0
    for i in range(l):
      c = c + j(i)
    return c

  if z is None:
    z = []

  n, D = numpy.shape(x)

  v = []

  if i is None:
    K = 0
    if z is None:
      z = []
    for j in range(l):
      f = cov(j)
      #K = K + f(f, hyp, x, z)
  else:
    if hi < len(v):
      vi = v(hi)
      #j = math.fsum()
      f = cov(vi)
      #K = f()
    else:
      raise AttributeError('Unknown hyperparameter')



def prod(cov, hyp, x=None, z=None, hi=None):
  raise NotImplementedError()

def add(cov, hyp, x=None, z=None, hi=None):
  raise NotImplementedError()

def scale(cov, hyp, x=None, z=None, hi=None):
  raise NotImplementedError()

def mask(cov, hyp, x=None, z=None, hi=None):
  raise NotImplementedError()

# Evaluates cov functions
def feval(fun, hyp=None, x=None, z=None, hi=None, dg=None):
  if not isinstance(fun, tuple):
    fun = (fun,)

  f = fun[0]
  #if f.__module__ == 'cov':
  if f.__module__ == 'sklearn.gpml.cov':
    if len(fun) > 1 and (f == cov.add or f == cov.mask or f == cov.prod or f == cov.scale or f == cov.sum):
      return f(fun[1], hyp, x, z, hi, dg)
    #elif f == cov.fitc or f == cov.maternIso or f == cov.poly or f == cov.ppIso:
    #  ...
    else:
      return f(hyp, x, z, hi, dg)
  else:
    raise AttributeError('Unknown function')
