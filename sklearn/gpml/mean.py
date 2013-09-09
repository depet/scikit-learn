import numpy

# Zero mean function. The mean function does not have any parameters.
#
# m(x) = 0
def zero(hyp, x=None, hi=None):
  if x is None:
    return '0'
  
  # derivative and mean
  return numpy.zeros((numpy.size(x,0),1))



def mask(mean, hyp, x=None, hi=None):
  raise NotImplementedError('')

def pow(mean, hyp, x=None, hi=None):
  raise NotImplementedError('')

def prod(mean, hyp, x=None, hi=None):
  raise NotImplementedError('')

def scale(mean, hyp, x=None, hi=None):
  raise NotImplementedError('')

def sum(mean, hyp, x=None, hi=None):
  raise NotImplementedError('')

# Evaluates mean functions
def feval(fun, hyp=None, x=None, hi=None):
  if not isinstance(fun, tuple):
    fun = (fun,)

  f = fun[0]
  #if f.__module__ == 'mean':
  if f.__module__ == 'sklearn.gpml.mean':
    if len(fun) > 1 and (f == mean.mask or f == mean.pow or f == mean.prod or f == mean.scale or f == mean.sum):
      return f(fun[1], hyp, x, hi)
    else:
      return f(hyp, x, hi)
  else:
    raise AttributeError('Unknown function')
