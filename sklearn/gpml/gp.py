import numpy
import scipy.optimize

from ..base import BaseEstimator

from . import cov
from . import inf
from . import lik
from . import mean
from . import util

class GP(BaseEstimator):
  """
  The GP model class.

  Parameters
  ----------
  x : double array_like
    An array with shape (n_samples, n_features) with the input at which
    observations were made.

  y : double array_like
    An array with shape (n_samples, ) with the observations of the
    scalar output to be predicted.

  hyp : double array like or dictionary, optional
    An array with shape (n_hyp, ) or dictionary containing keys 'mean', 
    'cov' and 'lik' and arrays with shape (n_hyp_KEY,) as 
    hyperparameter values.

  inffunc : string or callable, optional
    An inference function computing the (approximate) posterior for a 
    Gaussian process.
    Available built-in inference functions are::

      'exact', 'laplace', 'ep', 'vb', 'fitc', 'fitc_laplace', 'fitc_ep'

  meanfunc : string or callable, optional
    A mean function to be used by Gaussian process functions. 
    Available built-in simple mean functions are::

      'zero', 'one', 'const', 'linear'
      
    and composite mean functions are::

      'mask', 'pow', 'prod', 'scale', 'sum'

  covfunc : string or callable, optional
    A covariance function to be used by Gaussian process functions. 
    Available built-in simple covariance functions are::

      'const', 'lin', 'linard', 'linone', 'materniso', 'nnone',
      'noise', 'periodic', 'poly', 'ppiso', 'rqard', 'rqiso',
      'seard', 'seiso', 'seisou'
      
    and composite covariance functions are::

      'add', 'mask', 'prod', 'scale', 'sum'
      
    and special purpose (wrapper) covariance functions::

      'fitc'

  likfunc : string or callable, optional
    A likelihood function to be used by Gaussian process functions. 
    Available built-in likelihood functions are::

      'erf', 'logistic', 'uni', 'gauss', 'laplace', 'sech2', 't',
      'poisson'
    
    and composite likelihood functions are::

      'mix'

  Examples
  --------
  >>> import numpy as np
  >>> from sklearn.gpml import GP
  >>> X = np.array([[1., 3., 5., 6., 7., 8.]]).T
  >>> y = (X * np.sin(X)).ravel()
  >>> gp = GP(X, y)
  >>> gp.fit(X, y)                                      # doctest: +ELLIPSIS
  GP(beta0=None...
          ...
  >>> Xs = numpy.array([numpy.linspace(-1,10,101)]).T
  >>> mu, s2 = gp.predict(Xs)

  Notes
  -----
  The implementation is based on a translation of the GPML
  Matlab toolbox version 3.2, see reference [GPML32]_.

  References
  ----------

  .. [RN2013] `C.E. Rasmussen and H. Nickisch.  The GPML Toolbox version 3.2. (2013)`
      http://www.gaussianprocess.org/gpml/code/matlab/doc/manual.pdf

  .. [RW2006] `C.E. Rasmussen and C.K.I. Williams (2006). Gaussian Processes for 
      Machine Learning.  The MIT Press.`
      http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  """

  _inference_functions = {
    'exact': inf.exact}

  _mean_functions = {
    'zero': mean.zero}

  _covariance_functions = {
    'seard': cov.seArd}

  _likelihood_functions = {
    'gauss': lik.gauss}
    
  __prnt_counter = 0


  def __init__(self, x, y, hyp=None, inffunc=inf.exact, meanfunc=mean.zero, covfunc=cov.seArd, likfunc=lik.gauss):
    self.x, self.y, self.N, self.D = self.__setData(x, y)
    
    self.inff = self.__setFunc(inffunc, 'inference')
    self.meanf = self.__setFunc(meanfunc, 'mean')
    self.covf = self.__setFunc(covfunc, 'covariance')
    self.likf = self.__setFunc(likfunc, 'likelihood')
    
    self.hyp = self.__setHyp(hyp)


  def fit(self, x, y, hyp0=None, maxiter=200):
    """
    The GP model fitting method.

    Parameters
    ----------
    x : double array_like
      An array with shape (n_samples, n_features) with the input at which
      observations were made.

    y : double array_like
      An array with shape (n_samples, ) with the observations of the
      scalar output to be predicted.

    hyp0 : double array like or dictionary, optional
      An array with shape (n_hyp, ) or dictionary containing keys 'mean', 
      'cov' and 'lik' and arrays with shape (n_hyp_KEY,) as initial 
      hyperparameter values.

    Returns
    -------
    gp : self
      A fitted GP model object awaiting data to perform
      predictions.
    """

    self.x, self.y, self.N, self.D = self.__setData(x, y, False)
    if hyp0 is None:
      hyp0 = self.hyp
    else:
      hyp0 = self.__setHyp(hyp0)
    hyp0 = numpy.concatenate((numpy.reshape(hyp0['mean'],(-1,)), numpy.reshape(hyp0['cov'],(-1,)), numpy.reshape(hyp0['lik'],(-1,))))
    self.__prnt_counter = 0
    res = scipy.optimize.minimize(self._gp, hyp0, (self.inff, self.meanf, self.covf, self.likf, x, y), method='Newton-CG', jac=True, callback=self.__prnt, options={'maxiter': maxiter})
    hyp = res.x
    self.hyp = self.__setHyp(hyp)
    return self

  def __prnt(self, x):
    self.__prnt_counter += 1
    nlZ, dnlZ = self._gp(x, self.inff, self.meanf, self.covf, self.likf, self.x, self.y)
    print 'Iteration %d; Value %f' % (self.__prnt_counter, nlZ)


  def predict(self, xs, ys=None, batch_size=None, nargout=2):
    """
    This function evaluates the GP model at x.

    Parameters
    ----------
    xs : array_like
      An array with shape (n_eval, n_features) giving the point(s) at
      which the prediction(s) should be made.

    ys : array_like, optional
      An array with shape (n_eval, ) giving the real targets at ...
      Default assumes ys = None and evaluates only the mean
      prediction.

    batch_size : integer, optional
      An integer giving the maximum number of points that can be
      evaluated simultaneously (depending on the available memory).
      Default is None so that all given points are evaluated at the same
      time.

    Returns
    -------
    mu : array_like
      An array with shape (n_eval, ) with the mean value of prediction at xs.

    s2 : array_like
      An array with shape (n_eval, ) with the variance of prediction at xs.
    """

    if numpy.size(xs, 1) != self.D:
      if numpy.size(xs,0) == self.D:
        xs = xs.T
      else:
        raise AttributeError('Dimension of the test inputs (xs) disagree with dimension of the train inputs (x).')
    res = self._gp(self.hyp, self.inff, self.meanf, self.covf, self.likf, self.x, self.y, xs, ys, nargout, batch_size=batch_size)
    if nargout <= 1:
      return res[0]
    elif nargout <= 6:
      return res[:nargout]
    else:
      return res[:6]


  def _gp(self, hyp, inff, meanf, covf, likf, x, y, xs=None, ys=None, nargout=None, post=None, hypdict=None, batch_size=None):
    if nargout is None:
      nargout = 2

    if not isinstance(meanf, tuple):
      meanf = (meanf,)
    if not isinstance(covf, tuple):
      covf = (covf,)
    if not isinstance(likf, tuple):
      likf = (likf,)

    if isinstance(inff, tuple):
      inff = inff[0]

    D = numpy.size(x,1);
    
    if not isinstance(hyp, dict):
      if hypdict is None:
        hypdict = False
      shp = numpy.shape(hyp)
      hyp = numpy.reshape(hyp, (-1,1))
      ms = eval(mean.feval(meanf))
      hypmean = hyp[range(0,ms)]
      cs = eval(cov.feval(covf))
      hypcov = hyp[range(ms,cs)]
      ls = eval(lik.feval(likf))
      hyplik = hyp[range(cs,cs+ls)]
      hyp = {'mean': hypmean, 'cov': hypcov, 'lik': hyplik}
    else:
      if hypdict is None:
        hypdict = True
      if 'mean' not in hyp:
        hyp['mean'] = numpy.array([[]])
      if eval(mean.feval(meanf)) != numpy.size(hyp['mean']):
        raise AttributeError('Number of mean function hyperparameters disagree with mean function')

      if 'cov' not in hyp:
        hyp['cov'] = numpy.array([[]])
      if eval(cov.feval(covf)) != numpy.size(hyp['cov']):
        raise AttributeError('Number of cov function hyperparameters disagree with cov function')

      if 'lik' not in hyp:
        hyp['lik'] = numpy.array([[]])
      if eval(lik.feval(likf)) != numpy.size(hyp['lik']):
        raise AttributeError('Number of lik function hyperparameters disagree with lik function')

    # call the inference method
    try:
      # issue a warning if a classification likelihood is used in conjunction with
      # labels different from +1 and -1
      if lik == lik.erf or lik == lik.logistic:
        uy = numpy.unique(y)
        if numpy.any(~(uy == -1) & ~(uy == 1)):
          print 'You try classification with labels different from {+1,-1}'
      # compute marginal likelihood and its derivatives only if needed
      if xs is not None:
        if post is None:
          post = inff(hyp, meanf, covf, likf, x, y, nargout=1)
      else:
        if nargout == 1:
          post, nlZ = inff(hyp, meanf, covf, likf, x, y, nargout=2)
        else:
          post, nlZ, dnlZ = inff(hyp, meanf, covf, likf, x, y, nargout=3)
    except Exception, e:
      if xs is not None:
        raise Exception('Inference method failed [%s]' % (e,))
      else:
        print 'Warning: inference method failed [%s] .. attempting to continue' % (e,)
        if hypdict:
          dnlZ = {'cov': 0*hyp['cov'], 'mean': 0*hyp['mean'], 'lik': 0*hyp['lik']}
        else:
          if not hypdict:
            dnlZ = numpy.concatenate((0*numpy.reshape(hyp['mean'],(-1,1)), 0*numpy.reshape(hyp['cov'],(-1,1)), 0*numpy.reshape(hyp['lik'],(-1,1))))
            dnlZ = numpy.reshape(dnlZ, shp)
        return (numpy.NaN, dnlZ)


    if xs is None:
      if nargout == 1:
        return nlZ
      else:
        if not hypdict:
          dnlZ = numpy.concatenate((numpy.reshape(dnlZ['mean'],(-1,1)), numpy.reshape(dnlZ['cov'],(-1,1)), numpy.reshape(dnlZ['lik'],(-1,1)))).T
          dnlZ = numpy.reshape(dnlZ, shp)
        if nargout == 2:
          return (nlZ, dnlZ)
        else:
          return (nlZ, dnlZ, post)
    else:
      alpha = post['alpha']
      L = post['L']
      sW = post['sW']

      if not True: #issparse(alpha)
        nz = 0
      else:
        nz = numpy.tile(numpy.array([[True]]), (numpy.size(alpha,0),1))

      # if L is not provided, we compute it
      if numpy.size(L) == 0:
        K = cov.feval(covf, hyp=hyp['cov'], x=x[nz[:,0],:])
        L = numpy.linalg.cholesky(numpy.eye(numpy.sum(nz))+numpy.dot(sW,sW.T)*K).T
        post['L'] = L # not in GPML, check if it is really needed

      Ltril = numpy.all(numpy.tril(L,-1)==0)
      ns = numpy.size(xs,0)
      if batch_size is None:
        batch_size = 1000
      nact = 0

      # allocate memory
      ymu = numpy.zeros((ns,1))
      ys2 = numpy.zeros((ns,1))
      fmu = numpy.zeros((ns,1))
      fs2 = numpy.zeros((ns,1))
      lp  = numpy.zeros((ns,1))

      while nact < ns:
        id = range(nact, min(nact+batch_size, ns))
        kss = cov.feval(covf, hyp=hyp['cov'], x=xs[id,:], dg=True)
        Ks  = cov.feval(covf, hyp=hyp['cov'], x=x[nz[:,0],:], z=xs[id,:])
        ms = mean.feval(meanf, hyp=hyp['mean'], x=xs[id,:])
        N = numpy.size(alpha,1)
        Fmu = numpy.tile(ms,(1,N)) + numpy.dot(Ks.T,alpha[nz[:,0],:])
        fmu[id] = numpy.array([numpy.sum(Fmu,1)]).T/N

        if Ltril:
          V = numpy.linalg.solve(L.T, numpy.tile(sW,(1,len(id)))*Ks)
          fs2[id] = kss - numpy.array([numpy.sum(V*V,0)]).T
        else:
          fs2[id] = kss + numpy.array([numpy.sum(Ks*numpy.dot(L,Ks),0)]).T
        fs2[id] = numpy.maximum(fs2[id],0)
        Fs2 = numpy.tile(fs2[id],(1,N))

        if ys is None:
          Lp, Ymu, Ys2 = lik.feval(likf, hyp['lik'], y=numpy.array([[]]), mu=numpy.reshape(Fmu, (-1,1)), s2=numpy.reshape(Fs2, (-1,1)))
        else:
          Lp, Ymu, Ys2 = lik.feval(likf, hyp['lik'], y=numpy.tile(ys[id], (1,N)), mu=numpy.reshape(Fmu, (-1,1)), s2=numpy.reshape(Fs2, (-1,1)))

        lp[id]  = numpy.array([numpy.sum(numpy.reshape(Lp, (-1,N)),1)/N]).T
        ymu[id] = numpy.array([numpy.sum(numpy.reshape(Ymu, (-1,N)),1)/N]).T
        ys2[id] = numpy.array([numpy.sum(numpy.reshape(Ys2, (-1, N)),1)/N]).T
        nact = id[-1] + 1

      if ys is None:
        lp = numpy.array([[]])

      if nargout == 1:
        return ymu
      elif nargout == 2:
        return (ymu, ys2)
      elif nargout == 3:
        return (ymu, ys2, fmu)
      elif nargout == 4:
        return (ymu, ys2, fmu, fs2)
      elif nargout == 5:
        return (ymu, ys2, fmu, fs2, lp)
      else:
        return (ymu, ys2, fmu, fs2, lp, post)


  def __setData(self, x, y, init=True):
    if numpy.size(y, 0) > 1 and numpy.size(y, 1) > 1:
      raise AttributeError('Only one-dimensional targets (y) are supported.')
    y = numpy.reshape(y, (-1,1))
    N = numpy.size(y, 0)
    if numpy.size(x, 0) != N:
      if numpy.size(x, 1) != N:
        raise AttributeError('Number of inputs (x) and targets (y) must be the same.')
      else:
        x = x.T
    D = numpy.size(x, 1)
    if not init:
      if D != self.D:
        raise AttributeError('Input data (x) dimension disagree with hyperparameters.')
    return (x, y, N, D)


  def __setHyp(self, hyp):
    D = self.D
    
    ms = eval(mean.feval(self.meanf))
    cs = eval(cov.feval(self.covf))
    ls = eval(lik.feval(self.likf))

    if hyp is None:
      hyp = {}
      hyp['mean'] = numpy.zeros((ms,1))
      hyp['cov'] = numpy.zeros((cs,1))
      hyp['lik'] = numpy.zeros((ls,1))
    elif isinstance(hyp, dict):
      if 'mean' not in hyp:
        hyp['mean'] = numpy.array([[]])
      if ms != numpy.size(hyp['mean']):
        raise AttributeError('Number of mean function hyperparameters disagree with mean function.')
      if 'cov' not in hyp:
        hyp['cov'] = numpy.array([[]])
      if cs != numpy.size(hyp['cov']):
        raise AttributeError('Number of cov function hyperparameters disagree with cov function.')
      if 'lik' not in hyp:
        hyp['lik'] = numpy.array([[]])
      if ls != numpy.size(hyp['lik']):
        raise AttributeError('Number of lik function hyperparameters disagree with lik function.')
    elif isinstance(hyp, numpy.ndarray):
      hyp = numpy.reshape(hyp, (-1,1))

      if ms + cs + ls == numpy.size(hyp, 0):
        hypmean = hyp[range(0,ms)]
        hypcov = hyp[range(ms,cs)]
        hyplik = hyp[range(cs,cs+ls)]
        hyp = {'mean': hypmean, 'cov': hypcov, 'lik': hyplik}
      else:
        raise AttributeError('Number of hyperparameters disagree with functions.')
    else:
      raise AttributeError('Unsupported type of hyperparameters.')

    return hyp


  def __setFunc(self, f, ftype, lower=False):
    if ftype == 'inference':
      fs = self._inference_functions
      m = 'sklearn.gpml.inf'
    elif ftype == 'mean':
      fs = self._mean_functions
      m = 'sklearn.gpml.mean'
    elif ftype == 'covariance':
      fs = self._covariance_functions
      m = 'sklearn.gpml.cov'
    elif ftype == 'likelihood':
      fs = self._covariance_functions
      m = 'sklearn.gpml.lik'
    else:
      raise AttributeError('Unknown function type.')

    if not lower and not isinstance(f, tuple):
      f = (f,)

    resf = ()
    for fp in f:
      if isinstance(fp, basestring):
        if fp.lower() in fs:
          fp = fs[fp]
        else:
          raise AttributeError('Unknown %s function.' % ftype)
      elif isinstance(fp, tuple) and ftype != 'inference':
        fp = self.__setFunc(fp, ftype, True)
      elif hasattr(fp, '__call__'):
        if fp.__module__ != m:
          raise AttributeError('%s function not from %s module.' % (ftype.capitalize(), m))
      else:
        raise AttributeError('Unknown %s function type.' % ftype)
      resf = resf + (fp,)
      if ftype == 'inference':
        return fp

    return resf
