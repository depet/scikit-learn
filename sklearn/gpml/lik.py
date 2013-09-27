import numpy
import scipy.special

def gauss(hyp, y=None, mu=None, s2=None, inf=None, hi=None, nargout=None):
  """
  Gaussian likelihood function for regression. The expression for the 
  likelihood is 
    likGauss(t) = exp(-(t-y)^2/2*sn^2) / sqrt(2*pi*sn^2),
  where y is the mean and sn is the standard deviation.
 
  The hyperparameters are:
 
  hyp = [  log(sn)  ]
 
  Several modes are provided, for computing likelihoods, derivatives and moments
  respectively, see likFunctions.m for the details. In general, care is taken
  to avoid numerical issues when the arguments are extreme.
  """
  if mu is None:
    return '1'

  sn2 = numpy.exp(2*hyp)
  
  if inf is None:
    if numpy.size(y) == 0:
      y = numpy.zeros(numpy.shape(mu))
    
    if s2 is not None and numpy.linalg.norm(s2) > 0: # s2==0?
      out = gauss(hyp, y, mu, s2, 'ep')
      lp = out[0]
    else:
      lp = -(y-mu)**2/sn2/2-numpy.log(2*numpy.pi*sn2)/2
      s2 = 0
    if nargout == 1:
      return lp
    elif nargout == 2:
      return (mu, mu)
    else:
      return (lp, mu, s2 + sn2)
  else:
    if inf == 'laplace':
      if hi is None:
        if nargout is None:
          nargout = 4
        if numpy.size(y) == 0:
          y = 0
        ymmu = y-mu
        lp = -numpy.power(ymmu,2)/(2*sn2) - numpy.log(2*numpy.pi*sn2)/2
        res = lp
        if nargout > 1:
          dlp = ymmu/sn2
          res = (lp, dlp)
        if nargout > 2:
          d2lp = -numpy.ones(numpy.shape(ymmu))/sn2
          res += (d2lp,)
        if nargout > 3:
          d3lp = numpy.zeros(numpy.shape(ymmu))
          res += (d3lp)
      else:
        if nargout is None:
          nargout = 3
        lp_dhyp = numpy.power(y-mu,2)/sn2 - 1
        res = lp
        if nargout > 1:
          dlp_dhyp = 2*(mu-y)/sn2
          res = (lp, dlp_dhyp)
        if nargout > 2:
          d2lp_dhyp = 2*numpy.ones(numpy.shape(mu))/sn2
          res += (d2lp_dhyp,)
      return res
    elif inf == 'ep':
      if hi is None:
        if nargout is None:
          nargout = 3
        lZ = -(y-mu)**2/(sn2+s2)/2 - numpy.log(2*numpy.pi*(sn2+s2))/2
        dlZ  = (y-mu)/(sn2+s2)
        d2lZ = -1./(sn2+s2)
        if nargout == 1:
          return lZ
        elif nargout == 2:
          return (lZ, dlZ)
        else:
          return (lZ, dlZ, d2lZ)
      else:
        if nargout is None:
          nargout = 1
        dlZhyp = ((y-mu)**2/(sn2+s2)-1)/(1+s2/sn2)
        if nargout == 1:
          return dlZhyp
        else:
          res = (dlZhyp,)
          for i in range(2,nargout):
            res += (None,)
          return res
            
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


def erf(hyp, y=None, mu=None, s2=None, inf=None, hi=None, nargout=None):
  """
  Error function or cumulative Gaussian likelihood function for binary
  classification or probit regression. The expression for the likelihood is 
    likErf(t) = (1+erf(t/sqrt(2)))/2 = normcdf(t).
  
  Several modes are provided, for computing likelihoods, derivatives and moments
  respectively. In general, care is taken to avoid numerical issues when the 
  arguments are extreme.
  """
  if mu is None:
    return '0'

  if y is not None:
    if numpy.size(y) == 0:
      y = numpy.array([[1]])
    else:
      y = numpy.sign(y)
      y[y==0] = 1
  else:
    y = numpy.array([[1]])

  # prediction mode if inf is not present
  if inf is None:
    y = y*numpy.ones(numpy.shape(mu))
    if s2 is not None and numpy.linalg.norm(s2) > 0: # s2==0?
      lp = erf(hyp, y, mu, s2, 'ep', nargout=1)
      p = numpy.exp(lp)
    else:
      p, lp = __cumGauss(y,mu,nargout=2)
    if nargout is None:
      nargout = 3
    res = lp
    if nargout > 1:
      ymu = 2*p-1
      res = (lp, ymu)
      if nargout > 2:
        ys2 = 4*p*(1-p)
        res += (ys2,)
    return res
  else:
    # TODO: TEST
    if inf == 'laplace':
      # no derivative mode
      if hi is None:
        f = mu
        yf = y*f                                   # product latents and labels
        p, lp = __cumGauss(y, f, nargout=2)
        res = lp
        # derivative of log likelihood
        if nargout > 1:
          n_p = __gauOverCumGauss(yf, p)
          dlp = y*n_p                              # derivative of log likelihood
          res = (lp, dlp)
          # 2nd derivative of log likelihood
          if nargout > 2:
            d2lp = -numpy.power(n_p,2) - yf*n_p
            res += (d2lp,)
            # 3rd derivative of log likelihood
            if nargout > 3:
              d3lp = 2*y*numpy.power(n_p,3) + 3*f*numpy.power(n_p,2) + y*(numpy.power(f,2)-1)*n_p
              res += (d3lp,)
        return res
      # derivative mode
      else:
        return numpy.array([[]])
    elif inf == 'ep':
      if hi is None:
        if nargout is None:
          nargout = 3
        z = mu/numpy.sqrt(1+s2)
        # log part function
        junk, lZ = __cumGauss(y,z,nargout=2)
        res = lZ
        if numpy.size(y) > 0:
          z = z*y
        if nargout > 1:
          if numpy.size(y) == 0:
            y = 1
          n_p = __gauOverCumGauss(z,numpy.exp(lZ))
          # 1st derivative wrt mean
          dlZ = y*n_p/numpy.sqrt(1+s2)
          res = (lZ,dlZ)
          if nargout > 2:
            # 2nd derivative wrt mean
            d2lZ = -n_p*(z+n_p)/(1+s2)
            res += (d2lZ,)
        return res
      else:
        return numpy.array([[]])
    elif inf == 'vb':
      a = 0
    else:
      raise AttributeError('Unknown inference')


def __cumGauss(y, f, nargout=1):
  # product of latents and labels
  if numpy.size(y) > 0:
    yf = y*f
  else:
    yf = f
  # likelihood
  p = (1+scipy.special.erf(yf/numpy.sqrt(2)))/2
  res = p
  # log likelihood
  if nargout > 1:
    lp = __logphi(yf,p)
    res = (p,lp)
  return res


def __logphi(z, p):
  """
  safe implementation of the log of phi(x) = \int_{-\infty}^x N(f|0,1) df
  logphi(z) = log(normcdf(z))
  """
  lp = numpy.zeros(numpy.shape(z))
  zmin = -6.2
  zmax = -5.5
  ok = z > zmax
  bd = z < zmin
  # interpolate between both of them
  ip = ~ok & ~bd
  # interpolate weights
  lam = 1/(1+numpy.exp(25*(1/2-(z[ip]-zmin)/(zmax-zmin))))
  lp[ok] = numpy.log(p[ok])
  # use lower and upper bound acoording to Abramowitz&Stegun 7.1.13 for z<0
  # lower -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+2   ) -z/sqrt(2) )
  # upper -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+4/pi) -z/sqrt(2) )
  # the lower bound captures the asymptotics
  lp[~ok] = -numpy.log(numpy.pi)/2 -numpy.power(z[~ok],2)/2 - numpy.log(numpy.sqrt(numpy.power(z[~ok],2)/2+2)-z[~ok]/numpy.sqrt(2))
  lp[ip] = (1-lam)*lp[ip] + lam*numpy.log(p[ip])
  return lp


def __gauOverCumGauss(f, p):
  """
  Safely compute Gaussian over cumulative Gaussian.
  """
  n_p = numpy.zeros(numpy.shape(f))

  # naive evaluation for large values of f
  ok = f>-5
  n_p[ok] = (numpy.exp(-numpy.power(f[ok],2)/2)/numpy.sqrt(2*numpy.pi)) / p[ok]
  
  # tight upper bound evaluation
  bd = f < -6
  n_p[bd] = numpy.sqrt(numpy.power(f[bd],2)/4+1)-f[bd]/2

  # linearly interpolate between both of them
  interp = ~ok & ~bd
  tmp = f[interp]
  lam = -5-f[interp]
  n_p[interp] = (1-lam)*(numpy.exp(-numpy.power(tmp,2)/2)/numpy.sqrt(2*numpy.pi))/p[interp] + lam*(numpy.sqrt(numpy.power(tmp,2)/4+1)-tmp/2)
  return n_p


def logistic(hyp, y=None, mu=None, s2=None, inf=None, hi=None, nargout=None):
  """
  Logistic function for binary classification or logit regression.
  The expression for the likelihood is 
    logistic(t) = 1/(1+exp(-t)).
  
  Several modes are provided, for computing likelihoods, derivatives and moments
  respectively. In general, care is taken to avoid numerical issues when the 
  arguments are extreme. The moments \int f^k logistic(y,f) N(f|mu,var) df
  are calculated via a cumulative Gaussian scale mixture approximation.
  """
  if mu is None:
    return '0'

  if y is not None:
    if numpy.size(y) == 0:
      y = numpy.array([[1]])
    else:
      y = numpy.sign(y)
      y[y==0] = 1
  else:
    y = numpy.array([[1]])
  
  # prediction mode if inf is not present
  if inf is None:
    y = y*numpy.ones(numpy.shape(mu))
    if s2 is not None and numpy.linalg.norm(s2) > 0: # s2==0?
      lp = logistic(hyp, y, mu, s2, 'ep', nargout=1)
    else:
      yf = y*mu
      lp = yf.copy()
      ok = -35<yf
      lp[ok] = -numpy.log(1+numpy.exp(-yf[ok]))
    if nargout is None:
      nargout = 3
    res = lp
    if nargout > 1:
      p = numpy.exp(lp)
      ymu = 2*p-1
      res = (lp, ymu)
      if nargout > 2:
        ys2 = 4*p*(1-p)
        res += (ys2,)
    return res
  else:
    # TODO: TEST
    if inf == 'laplace':
      # no derivative mode
      if hi is None:
        # product latents and labels
        f = mu
        yf = y*f
        s = -yf
        ps = numpy.maximum(0,s)
        # lp = -(log(1+exp(s)))
        lp = -(ps+numpy.log(numpy.exp(-ps) + numpy.exp(s-ps)))
        res = lp
        # first derivatives
        if nargout > 1:
          s = numpy.minimum(0,f)
          p = numpy.exp(s)/(numpy.exp(s) + numpy.exp(s-f)) # p = 1./(1+exp(-f))
          dlp = (y+1)/2.-p                       # derivative of log likelihood
          res = (lp,dlp)
          # 2nd derivative of log likelihood
          if nargout > 2:
            d2lp = -numpy.exp(2*s-f)/numpy.power(numpy.exp(s)+numpy.exp(s-f),2)
            res += (d2lp,)
            # 3rd derivative of log likelihood
            if nargout > 3:
              d3lp = 2*d2lp*(0.5-p)
              res += (d3lp)
        return res
      # derivative mode
      else:
        return numpy.array([[]])
    elif inf == 'ep':
      if hi is None:
        if nargout is None:
          nargout = 3
        y = y*numpy.ones(numpy.shape(mu))
        # likLogistic(t) \approx 1/2 + \sum_{i=1}^5 (c_i/2) erf(lam_i/sqrt(2)t)
        # approx coeffs lam_i and c_i
        lam = numpy.sqrt(2)*numpy.array([[0.44, 0.41, 0.40, 0.39, 0.36]])
        c = numpy.array([[1.146480988574439e+02, -1.508871030070582e+03, 2.676085036831241e+03, -1.356294962039222e+03, 7.543285642111850e+01]]).T
        lZc, dlZc, d2lZc = erf({'cov': numpy.array([[]]), 'lik': numpy.array([[]]), 'mean': numpy.array([[]])}, numpy.dot(y,numpy.ones((1,5))), numpy.dot(mu,lam), numpy.dot(s2,numpy.power(lam,2)), inf, nargout=3)
        # A=lZc, B=dlZc, d=c.*lam', lZ=log(exp(A)*c)
        lZ = __log_expA_x(lZc,c)
        # ((exp(A).*B)*d)./(exp(A)*c)
        dlZ = __expABz_expAx(lZc, c, dlZc, c*lam.T)
        # d2lZ = ((exp(A).*Z)*e)./(exp(A)*c) - dlZ.^2 where e = c.*(lam.^2)'
        d2lZ = __expABz_expAx(lZc, c, numpy.power(dlZc,2)+d2lZc, c*numpy.power(lam,2).T) - numpy.power(dlZ,2)
        # The scale mixture approximation does not capture the correct asymptotic
        # behavior; we have linear decay instead of quadratic decay as suggested
        # by the scale mixture approximation. By observing that for large values 
        # of -f*y ln(p(y|f)) of logistic likelihood is linear in f with slope y,
        # we are able to analytically integrate the tail region; there is no 
        # contribution to the second derivative
        
        # empirically determined bound at val==0
        val = numpy.abs(mu)-196./200.*s2-4.
        # interpolation weights
        lam = 1/(1+numpy.exp(-10*val))
        # apply the same to p(y|f) = 1 - p(-y|f)
        lZtail = numpy.minimum(s2/2-numpy.abs(mu),-0.1)
        dlZtail = -numpy.sign(mu)
        id = y*mu>0
        # label and mean agree
        lZtail[id] = numpy.log(1-numpy.exp(lZtail[id]))
        dlZtail[id] = 0
        # interpolate between scale mixture ..
        lZ = (1-lam)*lZ + lam*lZtail
        # .. and tail approximation
        dlZ = (1-lam)*dlZ + lam*dlZtail
        res = lZ
        if nargout > 1:
          res = (lZ,dlZ)
          if nargout > 2:
            res += (d2lZ,)
        return res
      else:
        return numpy.array([[]])
    elif inf == 'vb':
      a = 0
    else:
      raise AttributeError('Unknown inference')


def __log_expA_x(A,x):
  """
  Computes y = log( exp(A)*x ) in a numerically safe way by subtracting the
  maximal value in each row to avoid cancelation after taking the exp.
  """
  N = numpy.size(A,1)
  # number of columns, max over columns
  maxA = numpy.reshape(numpy.max(A,1),(-1,1))
  # exp(A) = exp(A-max(A))*exp(max(A))
  return numpy.log(numpy.dot(numpy.exp(A-numpy.dot(maxA,numpy.ones((1,N)))),x)) + maxA


def __expABz_expAx(A,x,B,z):
  """
  Computes y = ( (exp(A).*B)*z ) ./ ( exp(A)*x ) in a numerically safe way
  The function is not general in the sense that it yields correct values for
  all types of inputs. We assume that the values are close together.
  """
  # number of columns, max over columns
  N = numpy.size(A,1)
  maxA = numpy.reshape(numpy.max(A,1),(-1,1))
  # subtract maximum value
  A = A - numpy.dot(maxA,numpy.ones((1,N)))
  return numpy.dot(numpy.exp(A)*B,z) / numpy.dot(numpy.exp(A),x)


def mix():
  raise NotImplementedError('')

# Evaluates lik functions
def feval(fun, hyp=None, y=None, mu=None, s2=None, inff=None, hi=None, nargout=None):
  if not isinstance(fun, tuple):
    fun = (fun,)

  f = fun[0]
  if f.__module__ == 'sklearn.gpml.lik':
    if len(fun) > 1 and f == lik.mix:
      return f(fun[1], hyp, y, mu, s2, inff, hi, nargout)
    else:
      return f(hyp, y, mu, s2, inff, hi, nargout)
  else:
    raise AttributeError('Unknown function')
