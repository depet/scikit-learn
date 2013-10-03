"""
Covariance functions to be use by Gaussian process functions. There are two
different kinds of covariance functions:: simple and composite::

simple mean functions::

  const      - covariance for constant functions
  lin        - linear covariance function
  linArd     - linear covariance function with ARD
  linOne     - linear covariance function with bias
  maternIso  - Matern covariance function with nu=1/2, 3/2 or 5/2
  nnOne      - neural network covariance function
  noise      - independent covariance function (i.e. white noise)
  periodic   - smooth periodic covariance function (1d) with unit period
  poly       - polynomial covariance function
  ppIso      - piecewise polynomial covariance function (compact support)
  rqArd      - rational quadratic covariance function with ARD
  rqIso      - isotropic rational quadratic covariance function
  seArd      - squared exponential covariance function with ARD
  seIso      - isotropic squared exponential covariance function
  seIsoU     - as above but without latent scale
  
composite covariance functions::

  scale     - scaled version of a covariance function
  prod      - products of covariance functions
  sum       - sums of covariance functions
  add       - additive covariance function
  mask      - mask some dimensions of the data

special purpose (wrapper) covariance functions::

  fitc      - to be used in conjunction with inf.fitc for large scale
              regression problems; any covariance function can be wrapped
              by cov.fitc such that the FITC approximation is applicable
"""
# Author: Dejan Petelin <http://www.linkedin.com/in/dejanpetelin>
# Python implementation of the GPML MATLAB Toolbox, version 3.2
# License: see copyright

import numpy
import numpy.matlib

from . import util


def lin(hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Linear covariance function. The covariance function is parameterized as:

  k(x^p,x^q) = x^p'*x^q

  The are no hyperparameters:

  hyp = [ ]

  Note that there is no bias or scale term; use covConst to add these.
  """
  #report number of parameters
  if x is None:
    return '0'

  if z is None:
    z = numpy.array([[]])

  if dg is None:
    dg = False

  xeqz = numpy.size(z) == 0

  # compute inner products
  if dg:
    K = numpy.sum(x*x,1)
  else:
    if xeqz:                                             # symmetric matrix Kxx
      K = numpy.dot(x,x.T)
    else:                                               # cross covariances Kxz
      K = numpy.dot(x,z.T)

  if hi is not None:                                              # derivatives
    raise AttributeError('Unknown hyperparameter')

  return K


def linArd(hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Linear covariance function with Automatic Relevance Determination (ARD). The
  covariance function is parameterized as:

  k(x^p,x^q) = x^p'*inv(P)*x^q

  where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
  D is the dimension of the input space. The hyperparameters are:

  hyp = [ log(ell_1)
          log(ell_2)
           ..
          log(ell_D) ]

  Note that there is no bias term; use covConst to add a bias.
  """
  #report number of parameters
  if x is None:
    return 'D'

  if z is None:
    z = numpy.array([[]])

  if dg is None:
    dg = False

  xeqz = numpy.size(z) == 0

  ell = numpy.exp(hyp)
  n, D = numpy.shape(x)
  x = numpy.dot(x,numpy.diagflat(1./ell))

  # compute inner products
  if dg:
    K = numpy.sum(x*x,1)
  else:
    if xeqz:                                             # symmetric matrix Kxx
      K = numpy.dot(x,x.T)
    else:                                               # cross covariances Kxz
      z = numpy.dot(z,numpy.diagflat(1./ell))
      K = numpy.dot(x,z.T)

  if hi is not None:                                              # derivatives
    if hi > 0 and hi < D:
      if dg:
        K = -2*x[:,[i]]*x[:,[i]]
      else:
        if xeqz:
          K = -2*numpy.dot(x[:,[i]],x[:,[i]].T)
        else:
          K = -2*numpy.dot(x[:,[i]],z[:,[i]].T)
    else:
      raise AttributeError('Unknown hyperparameter')

  return K


def linOne(hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Linear covariance function with a single hyperparameter. The covariance
  function is parameterized as:

  k(x^p,x^q) = (x^p'*x^q + 1)/t2;

  where the P matrix is t2 times the unit matrix. The second term plays the
  role of the bias. The hyperparameter is:

  hyp = [ log(sqrt(t2)) ]
  """
  #report number of parameters
  if x is None:
    return '1'

  if z is None:
    z = numpy.array([[]])

  if dg is None:
    dg = False

  xeqz = numpy.size(z) == 0

  it2 = numpy.exp(-2*hyp)

  # compute inner products
  if dg:
    K = numpy.sum(x*x,1)
  else:
    if xeqz:                                             # symmetric matrix Kxx
      K = numpy.dot(x,x.T)
    else:                                               # cross covariances Kxz
      K = numpy.dot(x,z.T)

  if hi is None:                                                  # covariances
    K = it2*(1+K)
  else:                                                           # derivatives
    if hi == 0:
      K = -2*it2*(1+K)
    else:
      raise AttributeError('Unknown hyperparameter')

  return K


def periodic(hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Stationary covariance function for a smooth periodic function, 
  with period p:
  
  k(x^p,x^q) = sf2 * exp(-2*sin^2(pi*||(x^p - x^q)||/p)/ell^2)
  
  where the hyperparameters are:
  
  hyp = [ log(ell)
          log(p)
          log(sqrt(sf2)) ]
  """
  #report number of parameters
  if x is None:
    return '3'
  
  if z is None:
    z = numpy.array([[]])
    
  if dg is None:
    dg = False
  
  xeqz = numpy.size(z) == 0
  
  n = numpy.size(x,0);
  ell = numpy.exp(hyp[0])
  p = numpy.exp(hyp[1])
  sf2 = numpy.exp(2*hyp[2])

  # precompute squared distances
  if dg:
    K = numpy.zeros((numpy.size(x,0),1))
  else:
    if xeqz:
      K = numpy.sqrt(util.sq_dist(x.T))
    else:
      K = numpy.sqrt(util.sq_dist(x.T,z.T))

  K = numpy.pi*K/p
  if hi is None:
    K = numpy.sin(K)/ell
    K = K*K
    K = sf2*numpy.exp(-2*K)
  else:
    if hi == 0:
      K = numpy.sin(K)/ell
      K = K*K
      K = 4*sf2*numpy.exp(-2*K)*K
    elif hi == 1:
      R = numpy.sin(K)/ell
      K = 4*sf2/ell*numpy.exp(-2*R*R)*R*numpy.cos(K)*K
    elif hi == 2:
      K = numpy.sin(K)/ell
      K = K*K
      K = 2*sf2*numpy.exp(-2*K)
    else:
      raise AttributeError('Unknown hyperparameter')

  return K


def rqIso(hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Rational Quadratic covariance function with isotropic distance measure.
  The covariance function is parameterized as:
  
  k(x^p,x^q) = sf2 * [1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha)]^(-alpha)
  
  where the P matrix is ell^2 times the unit matrix, sf2 is the signal 
  variance and alpha is the shape parameter for the RQ covariance. The 
  hyperparameters are:
  
  hyp = [ log(ell)
          log(sqrt(sf2))
          log(alpha)     ]
  """
  #report number of parameters
  if x is None:
    return '3'
  
  if z is None:
    z = numpy.array([[]])
    
  if dg is None:
    dg = False
  
  xeqz = numpy.size(z) == 0
  
  ell = numpy.exp(hyp[0])
  sf2 = numpy.exp(2*hyp[1])
  alpha = numpy.exp(hyp[2])

  # precompute squared distances
  if dg:
    D2 = numpy.zeros((numpy.size(x,0),1))
  else:
    if xeqz:
      D2 = util.sq_dist(x.T/ell)
    else:
      D2 = util.sq_dist(x.T/ell,z.T/ell)

  if hi is None:
    K = sf2*numpy.power(1+0.5*D2/alpha,-alpha)
  else:
    if hi == 0:
      K = sf2*numpy.power(1+0.5*D2/alpha,-alpha-1)*D2
    elif hi == 1:
      K = 2*sf2*numpy.power(1+0.5*D2/alpha,-alpha)
    elif hi == 2:
      K = 1+0.5*D2/alpha
      K = sf2*numpy.power(K,-alpha)*(0.5*D2/K - alpha*numpy.log(K))
    else:
      raise AttributeError('Unknown hyperparameter')

  return K


def rqArd(hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Rational Quadratic covariance function with Automatic Relevance Determination
  (ARD) distance measure. The covariance function is parameterized as:

  k(x^p,x^q) = sf2 * [1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha)]^(-alpha)

  where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
  D is the dimension of the input space, sf2 is the signal variance and alpha
  is the shape parameter for the RQ covariance. The hyperparameters are:

  hyp = [ log(ell_1)
          log(ell_2)
           ..
          log(ell_D)
          log(sqrt(sf2))
          log(alpha) ]
  """
  #report number of parameters
  if x is None:
    return 'D+2'
  
  if z is None:
    z = numpy.array([[]])
    
  if dg is None:
    dg = False
  
  xeqz = numpy.size(z) == 0
  
  n, D = numpy.shape(x)

  ell = numpy.exp(hyp[0:D])
  sf2 = numpy.exp(2*hyp[D])
  alpha = numpy.exp(hyp[D+1])

  # precompute squared distances
  if dg:                                                           # vector kxx
    D2 = numpy.zeros((numpy.size(x,0),1))
  else:
    if xeqz:                                             # symmetric matrix Kxx
      D2 = util.sq_dist(numpy.dot(numpy.diagflat(1./ell),x.T))
    else:                                               # cross covariances Kxz
      D2 = util.sq_dist(numpy.dot(numpy.diagflat(1./ell),x.T), numpy.dot(numpy.diagflat(1./ell),z.T))

  if hi is None:                                                  # covariances
    K = sf2*numpy.power(1+0.5*D2/alpha, -alpha)
  else:                                                           # derivatives
    if hi >= 0 and hi < D:                             # length scale parameter
      if dg:
        K = D2*0
      else:
        if xeqz:
          K = sf2*numpy.power(1+0.5*D2/alpha, -alpha-1)*util.sq_dist(x[:,[i].T/ell[i])
        else:
          K = sf2*numpy.power(1+0.5*D2/alpha, -alpha-1)*util.sq_dist(x[:,[i]].T/ell[i], z[:,[i]].T/ell[i])
    elif hi == D:                                         # magnitude parameter
      K = 2*sf2*numpy.power(1+0.5*D2/alpha, -alpha)
    elif hi == D+1:
      K = 1+0.5*D2/alpha
      K = sf2*numpy.power(K, -alpha)*(0.5*D2/K-numpy.dot(alpha,numpy.log(K)))
    else:
      raise AttributeError('Unknown hyperparameter')

  return K


def seArd(hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Squared Exponential covariance function with Automatic Relevance Detemination
  (ARD) distance measure. The covariance function is parameterized as:
  
  k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
  
  where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
  D is the dimension of the input space and sf2 is the signal variance. The
  hyperparameters are:
  """
  #report number of parameters
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


def seIso(hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Squared Exponential covariance function with isotropic distance measure. 
  The covariance function is parameterized as:
  
  k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
  
  where the P matrix is ell^2 times the unit matrix and sf2 is the signal 
  variance. The hyperparameters are:
  
  hyp = [ log(ell)
          log(sf)  ]
  """
  #report number of parameters
  if x is None:
    return '2'
  
  if z is None:
    z = numpy.array([[]])
    
  if dg is None:
    dg = False
  
  xeqz = numpy.size(z) == 0

  n, D = numpy.shape(x)
  
  ell = numpy.exp(hyp[0])      # characteristic length scale
  sf2 = numpy.exp(2*hyp[1])        # signal variance

  # precompute squared distances
  if dg:
    K = numpy.zeros((numpy.size(x,0),1))
  else:
    if xeqz:
      K = util.sq_dist(x.T/ell)
    else:
      K = util.sq_dist(x.T/ell,z.T/ell)

  if hi is None:
    K = sf2*numpy.exp(-K/2)
  else:
    if hi == 0:
      K = sf2*numpy.exp(-K/2)*K
    elif hi == 1:
      K = 2*sf2*numpy.exp(-K/2)
    else:
      raise AttributeError('Unknown hyperparameter')

  return K


def seIsoU(hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Squared Exponential covariance function with isotropic distance measure
  with unit magnitude. The covariance function is parameterized as:
  
  k(x^p,x^q) = exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
  
  where the P matrix is ell^2 times the unit matrix. The hyperparameters 
  are:
  
  hyp = [ log(ell) ]
  """
  #report number of parameters
  if x is None:
    return '1'
  
  if z is None:
    z = numpy.array([[]])
    
  if dg is None:
    dg = False
  
  xeqz = numpy.size(z) == 0

  n, D = numpy.shape(x)
  
  ell = numpy.exp(hyp[0])      # characteristic length scale

  # precompute squared distances
  if dg:
    K = numpy.zeros((numpy.size(x,0),1))
  else:
    if xeqz:
      K = util.sq_dist(x.T/ell)
    else:
      K = util.sq_dist(x.T/ell,z.T/ell)

  if hi is None:
    K = numpy.exp(-K/2)
  else:
    if hi == 0:
      K = numpy.exp(-K/2)*K
    else:
      raise AttributeError('Unknown hyperparameter')

  return K


def noise(hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Independent covariance function, ie "white noise", with specified variance.
  The covariance function is specified as:
  
  k(x^p,x^q) = s2 * \delta(p,q)
  
  where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
  which is 1 iff p=q and zero otherwise. Two data points p and q are considered
  equal if their norm is less than 1e-9. The hyperparameter is
  
  hyp = [ log(sqrt(s2)) ]
  """
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


def sum(covf, hyp, x=None, z=None, hi=None, dg=None):
  """
  Compose a covariance function as the sum of other covariance
  functions. This function doesn't actually compute very much on its own, it
  merely does some bookkeeping, and calls other covariance functions to do the
  actual work.
  """
  if covf is None:
    raise AttributeError('Covariance functions to be summed must be defined.')
  else:
    l = len(covf)
    if l == 0:
      raise AttributeError('At least one covariance function to be summed must be defined.')

  # iterate over covariance functions and collect number of hyperparameters
  j = []
  for i in range(l):
    f = covf[i]
    j.append(feval(f))

  # if there is no data, return number of hyperparameters
  if x is None:
    t = j[0]
    for i in range(1, l):
      t = t + '+' + j[i]
    return t

  if z is None:
    z = z = numpy.array([[]])

  n, D = numpy.shape(x)

  # v vector indicates to which mean parameters belong
  v = numpy.array([], dtype=numpy.dtype(numpy.int32))
  for i in range(l):
    v = numpy.append(v, numpy.tile(i, (eval(j[i]),)))

  if hi is None:
    K = 0
    # compute covariance by iteration over summand functions
    for i in range(l):
      f = covf[i]
      # accumulate covariances
      K = K + feval(f, hyp[v==i], x, z, None, dg)
  else:
    # derivative
    if hi < numpy.size(v):
      i = v[hi]
      hj = numpy.sum(v[0:hi]==i)
      f = covf[i]
      K = feval(f, hyp[v==i], x, z, hj, dg)
    else:
      raise AttributeError('Unknown hyperparameter')

  return K


def prod(covf, hyp=None, x=None, z=None, hi=None, dg=None):
  """
  Compose a covariance function as the product of other covariance
  functions. This function doesn't actually compute very much on its own, it
  merely does some bookkeeping, and calls other covariance functions to do the
  actual work.
  """
  if covf is None:
    raise AttributeError('Covariance functions to be multiplied must be defined.')
  else:
    l = len(covf)
    if l == 0:
      raise AttributeError('At least one covariance function to be multiplied must be defined.')

  # iterate over covariance functions and collect number of hyperparameters
  j = []
  for i in range(l):
    f = covf[i]
    j.append(feval(f))

  # if there is no data, return number of hyperparameters
  if x is None:
    t = j[0]
    for i in range(1, l):
      t = t + '+' + j[i]
    return t

  if z is None:
    z = z = numpy.array([[]])

  n, D = numpy.shape(x)

  # v vector indicates to which mean parameters belong
  v = numpy.array([], dtype=numpy.dtype(numpy.int32))
  for i in range(l):
    v = numpy.append(v, numpy.tile(i, (eval(j[i]),)))

  if hi is None:
    K = 1
    # compute covariance by iteration over summand functions
    for i in range(l):
      f = covf[i]
      # accumulate covariances
      K = K * feval(f, hyp[v==i], x, z, None, dg)
  else:
    # derivative
    if hi < numpy.size(v):
      K = 1
      i = v[hi]
      hj = numpy.sum(v[0:hi]==i)
      for j in range(l):
        f = covf[j]
        if j == i:
          K = K * feval(f, hyp[v==j], x, z, hj, dg)
        else:
          K = K * feval(f, hyp[v==j], x, z, None, dg)
    else:
      raise AttributeError('Unknown hyperparameter')

  return K




def add(covf, hyp=None, x=None, z=None, hi=None):
  raise NotImplementedError()

def scale(covf, hyp=None, x=None, z=None, hi=None):
  raise NotImplementedError()

def mask(covf, hyp=None, x=None, z=None, hi=None):
  raise NotImplementedError()




def fitc(covf, xu, hyp=None, x=None, z=None, hi=None, dg=None, nargout=1):
  """
  Covariance function to be used together with the FITC approximation.
  
  The function allows for more than one output argument and does not respect the
  interface of a proper covariance function. In fact, it wraps a proper
  covariance function such that it can be used together with inf.fitc.
  Instead of outputing the full covariance, it returns cross-covariances between
  the inputs x, z and the inducing inputs xu as needed by inf.fitc.
  """
  if x is None:
    return feval(covf)

  if z is None:
    z = z = numpy.array([[]])

  if dg is None:
    dg = False
  
  xeqz = numpy.size(z) == 0

  if numpy.size(xu,1) != numpy.size(x,1):
    raise AttributeError('Dimensionality of inducing inputs must match training inputs.')

  if hi is None:
    # covariances
    if dg:
      return feval(covf,hyp,x,dg=True)
    else:
      if xeqz:
        K = feval(covf,hyp,x,dg=True)
        r = K
        if nargout >= 2:
          Kuu = feval(covf,hyp,xu)
          r = (K,Kuu)
        if nargout >= 3:
          Ku  = feval(covf,hyp,xu,x)
          r = (K,Kuu,Ku)
        return r
      else:
        return feval(covf,hyp,xu,z)
  else:
    # derivatives
    if dg:
      return feval(covf,hyp,x,hi=hi,dg=True)
    else:
      if xeqz:
        K = feval(covf,hyp,x,hi=hi,dg=True)
        r = K
        if nargout >= 2:
          Kuu = feval(covf,hyp,xu,hi=hi)
          r = (K,Kuu)
        if nargout >= 3:
          Ku  = feval(covf,hyp,xu,x,hi=hi)
          r = (K,Kuu,Ku)
        return r
      else:
        return feval(covf,hyp,xu,z,hi)



def feval(fun, hyp=None, x=None, z=None, hi=None, dg=None, nargout=None):
  """
  Evaluates covariance functions.
  """
  if not isinstance(fun, tuple):
    fun = (fun,)

  f = fun[0]
  if f.__module__ == __name__:
    if len(fun) > 1 and (f == add or f == mask or f == prod or f == scale or f == sum):
      return f(fun[1], hyp, x, z, hi, dg)
    elif f == fitc:
      if len(fun) < 3:
        raise AttributeError('FITC covariance function must contain pseudo inputs.')
      return f(fun[1], fun[2], hyp, x, z, hi, dg, nargout)
    #f == cov.maternIso or f == cov.poly or f == cov.ppIso:
    #  ...
    else:
      return f(hyp, x, z, hi, dg)
  else:
    raise AttributeError('Unknown covariance function.')
