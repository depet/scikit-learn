import numpy

# likGauss - Gaussian likelihood function for regression. The expression for the 
# likelihood is 
#   likGauss(t) = exp(-(t-y)^2/2*sn^2) / sqrt(2*pi*sn^2),
# where y is the mean and sn is the standard deviation.
#
# The hyperparameters are:
#
# hyp = [  log(sn)  ]
#
# Several modes are provided, for computing likelihoods, derivatives and moments
# respectively, see likFunctions.m for the details. In general, care is taken
# to avoid numerical issues when the arguments are extreme.
def gauss(hyp, y, mu=None, s2=None, inf=None, hi=None, nargout=None):
  if nargout is None:
    nargout = 3
    
  if mu is None:
    return '1'

  sn2 = numpy.exp(2*hyp)
  
  if inf is None:
    if numpy.size(y) == 0:
      y = numpy.zeros(numpy.shape(mu))
    
    if s2 is not None and numpy.linalg.norm(s2) > 0: # s2==0?
      out = gauss(hyp, y, mu, s2, 'infEP')
      lp = out[0]
    else:
      lp = -(y-mu)**2/sn2/2-numpy.log(2*numpy.pi*sn2)/2
      s2 = 0
    return (lp, mu, s2 + sn2)
  else:
    if inf == 'infLaplace':
      i = 0
#      if hi is None:
#        if numpy.size(y) == 0:
#          y = 0
#        ymmu = y-mu
#        lp = -ymmu.^2/(2*sn2) - log(2*pi*sn2)/2
#        dlp = ymmu/sn2
#        d2lp = -ones(size(ymmu))/sn2
#        d3lp = zeros(size(ymmu))
#        return (lp, dlp, d2lp, d3lp)
#      else:
#        lp_dhyp = (y-mu).^2/sn2 - 1
#        dlp_dhyp = 2*(mu-y)/sn2
#        d2lp_dhyp = 2*ones(size(mu))/sn2
#        return (lp_dhyp, dlp_dhyp, d2lp_dhyp)
    elif inf == 'infEP':
      if hi is None:
        lZ = -(y-mu)**2/(sn2+s2)/2 - numpy.log(2*numpy.pi*(sn2+s2))/2
        dlZ  = (y-mu)/(sn2+s2)
        d2lZ = -1./(sn2+s2)
        return (lZ, dlZ, d2lZ)
      else:
        dlZhyp = ((y-mu)**2/(sn2+s2)-1)/(1+s2/sn2)
        return (dlZhyp,)
#    elif inf == 'infVB':
#      if hi is None:
#        # variational lower site bound
#        # t(s) = exp(-(y-s)^2/2sn2)/sqrt(2*pi*sn2)
#        # the bound has the form: b*s - s.^2/(2*ga) - h(ga)/2 with b=y/ga
#        ga = s2
#        n = numel(ga)
#        b = y./ga
#        y = y.*ones(n,1)
#        db = -y./ga.^2
#        d2b = 2*y./ga.^3
#        h = zeros(n,1)
#        dh = h
#        d2h = h
#        id = ga(:)<=sn2+1e-8
#        h(id) = y(id).^2./ga(id) + log(2*pi*sn2)
#        h(~id) = Inf
#        dh(id) = -y(id).^2./ga(id).^2
#        d2h(id) = 2*y(id).^2./ga(id).^3
#        id = ga<0
#        h(id) = numpy.inf
#        dh(id) = 0
#        d2h(id) = 0
#        return (h, b, dh, db, d2h, d2b)
#      else:
#        ga = s2
#        n = numel(ga)
#        dhhyp = zeros(n,1)
#        dhhyp(ga(:)<=sn2) = 2
#        dhhyp(ga<0) = 0
#        return (dhhyp,)
    else:
      raise AttributeError('Unknown inference')


def mix():
  raise NotImplementedError('')

# Evaluates lik functions
def feval(fun, hyp=None, y=None, mu=None, s2=None, inff=None, hi=None, nargout=None):
  if not isinstance(fun, tuple):
    fun = (fun,)

  f = fun[0]
  #if f.__module__ == 'lik':
  if f.__module__ == 'sklearn.gpml.lik':
    if len(fun) > 1 and f == lik.mix:
      return f(fun[1], hyp, y, mu, s2, inff, hi, nargout)
    else:
      return f(hyp, y, mu, s2, inff, hi, nargout)
  else:
    raise AttributeError('Unknown function')
