"""
Mean functions to be use by Gaussian process functions. There are two
different kinds of mean functions:: simple and composite::

simple mean functions::

  zero      - zero mean function
  one       - one mean function
  const     - constant mean function
  linear    - linear mean function
  
composite covariance functions::

  scale     - scaled version of a mean function
  pow       - power of a mean function
  prod      - products of mean functions
  sum       - sums of mean functions
  mask      - mask some dimensions of the data
"""
# Author: Dejan Petelin <http://www.linkedin.com/in/dejanpetelin>
# Python implementation of the GPML MATLAB Toolbox, version 3.2
# License: see copyright

import numpy

def zero(hyp=None, x=None, hi=None):
  """
  Zero mean function. The mean function does not have any parameters.
  
  m(x) = 0
  """
  if x is None:
    return '0'
  
  # derivative and mean
  return numpy.zeros((numpy.size(x,0),1))


def one(hyp=None, x=None, hi=None):
  """
  One mean function. The mean function does not have any parameters.
  
  m(x) = 1
  """
  if x is None:
    return '0'
  
  if hi is None:
    # mean
    return numpy.ones((numpy.size(x,0),1))
  else:
    # derivative
    return numpy.zeros((numpy.size(x,0),1))


def const(hyp=None, x=None, hi=None):
  """
  Constant mean function. The mean function is parameterized as::
  
  m(x) = c
  
  The hyperparameter is::
  
  hyp = [ c ]
  """
  if x is None:
    return '1'

  if numpy.size(hyp) != 1:
    raise AttributeError('Exactly one hyperparameter needed.')

  c = numpy.reshape(hyp, (-1,))[0]
  if hi is None:
    # mean
    return c*numpy.ones((numpy.size(x,0),1))
  else:
    # derivative
    if hi == 0:
      return numpy.ones((numpy.size(x,0),1))
    else:
      return numpy.zeros((numpy.size(x,0),1))


def linear(hyp=None, x=None, hi=None):
  """
  Linear mean function. The mean function is parameterized as::
  
  m(x) = sum_i a_i * x_i
  
  The hyperparameters are::
  
  hyp = [ a_1
          a_2
          ...
          a_D]
  """
  if x is None:
    return 'D'

  n, D = numpy.shape(x)
  if numpy.shape(hyp) != (D,1):
    raise AttributeError('Exactly D hyperparameters needed.')
  
  a = hyp
  if hi is None:
    # mean
    return numpy.dot(x, a)
  else:
    # derivative
    if hi < D:
      return x[:,hi:hi+1]
    else:
      return numpy.zeros((n,1))


def scale(meanf, hyp=None, x=None, hi=None):
  """
  Compose a mean function as a scaled version of another one::
  
  m(x) = a * m_0(x)
  
  The hyperparameter is::
  
  hyp = [ a ]
  
  This function doesn't actually compute very much on its own, it merely does
  some bookkeeping, and calls other mean function to do the actual work.
  """
  if x is None:
    return feval(meanf, '+1')

  n, D = numpy.shape(x)
  a = numpy.reshape(hyp, (-1,))[0]
  if hi is None:
    # mean
    return a*feval(meanf,hyp[1:],x)
  else:
    # derivative
    if hi == 0:
      return feval(meanf,hyp[1:],x)
    else:
      return a*feval(meanf,hyp[1:],x,hi-1)


def pow(meanf, hyp=None, x=None, hi=None):
  """
  Compose a mean function as the power of another mean function::
  
  m(x) = m_0(x) ^ d
  
  The degree d has to be a strictly positive integer. It is defined in the 
  mean function as it is not to be optimized.
  
  This function doesn't actually compute very much on its own, it merely does
  some bookkeeping, and calls other mean function to do the actual work.
  """
  d = meanf[0]
  if not isinstance(d, (int, long, float)):
    raise AttributeError('First element of pow mean function must be a number.')
  d = max(numpy.abs(int(d)),1)
  meanf = meanf[1]

  if x is None:
    return feval(meanf)

  n, D = numpy.shape(x)
  if hi is None:
    # mean
    return numpy.power(feval(meanf,hyp,x), d)
  else:
    # derivative
    return (d*numpy.power(feval(meanf, hyp, x), (d-1))) * feval(meanf, hyp, x, hi);


def prod(meanf, hyp=None, x=None, hi=None):
  """
  Compose a mean function as the product of other mean functions::
  
  m(x) = prod_i m_i(x)
  
  This function doesn't actually compute very much on its own, it merely does
  some bookkeeping, and calls other mean function to do the actual work.
  """
  j = []
  for i in range(len(meanf)):
    f = meanf[i]
    j.append(feval(f))
  
  if x is None:
    t = j[0]
    for i in range(1, len(meanf)):
      t = t + '+' + j[i]
    return t

  n, D = numpy.shape(x)
  
  # v vector indicates to which mean parameters belong
  v = numpy.array([])
  for i in range(len(meanf)):
    v = numpy.append(v, numpy.tile(i, (eval(j[i]),)))

  r = numpy.ones((n,1))
  if hi is None:
    # compute mean by iteration over factor functions
    for i in range(len(meanf)):
      f = meanf[i]
      # accumulate means
      r = r * feval(f, hyp[v==i], x)
    return r
  else:
    # derivative
    if hi < numpy.size(v):
      i = v[hi]
      hj = numpy.sum(v[0:hi]==i)
      for j in range(len(meanf)):
        f = meanf[j]
        if j == i:
          # multiply derivative
          r = r * feval(f, hyp[v==j], x, hj)
        else:
          # multiply mean
          r = r * feval(f, hyp[v==j], x)
      return r
    else:
      return numpy.zeros((n,1))


def sum(meanf, hyp=None, x=None, hi=None):
  """
  Compose a mean function as the sum of other mean functions::
  
  m(x) = sum_i m_i(x)
  
  This function doesn't actually compute very much on its own, it merely does
  some bookkeeping, and calls other mean function to do the actual work.
  """
  if meanf is None:
    raise AttributeError('Mean functions to be summed must be defined.')
  else:
    l = len(meanf)
    if l == 0:
      raise AttributeError('At least one mean function to be summed must be defined.')


  # iterate over mean functions and collect number of hyperparameters
  j = []
  for i in range(len(meanf)):
    f = meanf[i]
    j.append(feval(f))
  
  # if there is no data, return number of hyperparameters
  if x is None:
    t = j[0]
    for i in range(1, l):
      t = t + '+' + j[i]
    return t

  n, D = numpy.shape(x)
  
  # v vector indicates to which mean parameters belong
  v = numpy.array([], dtype=numpy.dtype(numpy.int32))
  for i in range(len(meanf)):
    v = numpy.append(v, numpy.tile(i, (eval(j[i]),)))

  if hi is None:
    r = numpy.zeros((n,1))
    # compute mean by iteration over summand functions
    for i in range(len(meanf)):
      f = meanf[i]
      # accumulate means
      r = r + feval(f, hyp[v==i], x)
    return r
  else:
    # derivative
    if hi < numpy.size(v):
      i = v[hi]
      hj = numpy.sum(v[0:hi]==i)
      f = meanf[i]
      return feval(f, hyp[v==i], x, hj)
    else:
      return numpy.zeros((n,1))


def mask(meanf, hyp=None, x=None, hi=None):
  """
  Apply a mean function to a subset of the dimensions only. The subset can
  either be specified by a 0/1 mask, by a boolean mask or by an index set.
  
  This function doesn't actually compute very much on its own, it merely does
  some bookkeeping, and calls other mean function to do the actual work.
  """
  mask = numpy.fix(meanf[0])
  meanf = meanf[1]
  nh = feval(meanf)

  if numpy.max(mask) < 2 and numpy.size(mask) > 1:
    mask = numpy.nonzero(mask)[0]
  D = len(mask)
  if x is None:
    return str(eval(nh))

  if D > numpy.size(x,1):
    raise AttributeError('Size of masked data does not match the dimension of data.')
  if eval(nh) != numpy.size(hyp):
    raise AttributeError('Number of hyperparameters does not match size of masked data.')

  if hi is None:
    # mean
    return feval(meanf, hyp, x[:,mask])
  else:
    # derivative
    return feval(meanf, hyp, x[:,mask], hi)


def feval(fun, hyp=None, x=None, hi=None):
  """
  Evaluates mean functions.
  """
  if not isinstance(fun, tuple):
    fun = (fun,)

  f = fun[0]
  if f.__module__ == __name__:
    if len(fun) > 1 and (f == mask or f == pow or f == prod or f == scale or f == sum):
      return f(fun[1], hyp, x, hi)
    else:
      return f(hyp, x, hi)
  else:
    raise AttributeError('Unknown mean function')
