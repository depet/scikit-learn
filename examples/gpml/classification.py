"""
=========================================================================
GPML classification: exploiting the probabilistic output
=========================================================================

A binary classification, using two partially overlapping Gaussian sources of 
data in two dimensions.
"""
print(__doc__)

# Author: Dejan Petelin <http://www.linkedin.com/in/dejanpetelin>
# Licence: BSD 3 clause

import numpy

try:
  import matplotlib.pyplot as plt
  from matplotlib import cm
  matplotlib = True
except Exception, e:
  matplotlib = False

from sklearn import gpml

# generate data
n1 = 80
n2 = 40

S1 = numpy.eye(2)
S2 = numpy.array([[1, 0.95],[0.95, 1]])

m1 = numpy.array([[0.75, 0]]).T
m2 = numpy.array([[-0.75, 0]]).T

x1 = numpy.dot(numpy.linalg.cholesky(S1),numpy.random.randn(2, n1)) + m1
x2 = numpy.dot(numpy.linalg.cholesky(S2),numpy.random.randn(2, n2)) + m2

x = numpy.concatenate((x1, x2),1).T
y = numpy.concatenate((-numpy.ones((1,n1)), numpy.ones((1,n2))),1).T

t1, t2 = numpy.meshgrid(numpy.linspace(-4,4,81),numpy.linspace(-4,4,81))
t = numpy.concatenate((numpy.reshape(t1.T,(-1,1)),numpy.reshape(t2.T,(-1,1))),1)

# init GP model
hyp = {}
hyp['cov'] = numpy.array([[0, 0, 0]]).T
hyp['lik'] = numpy.array([[]])
hyp['mean'] = numpy.array([[]])

gp = gpml.GP(x, y, hyp, gpml.inf.ep, gpml.mean.zero, gpml.cov.seArd, gpml.lik.erf)
gp = gp.fit(x, y, maxiter=50)

a, b, c, d, lp = gp.predict(t, numpy.ones((numpy.size(t),1)), nargout=5)

if matplotlib:
  f = plt.figure()
  plt.plot(numpy.reshape(x[:,0],(-1,1))[y<=0], numpy.reshape(x[:,1],(-1,1))[y<=0], 'b.', markersize=12)
  plt.plot(numpy.reshape(x[:,0],(-1,1))[y>0], numpy.reshape(x[:,1],(-1,1))[y>0], 'r.', markersize=12)

  cax = plt.imshow(numpy.flipud(numpy.reshape(numpy.exp(lp),(81,81)).T), cmap=plt.cm.gray_r, alpha=0.8, extent=(-4, 4, -4, 4))
  norm = plt.matplotlib.colors.Normalize(vmin=0.1, vmax=0.9)
  cb = plt.colorbar(cax, ticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], norm=norm)

  cs = plt.contour(t1, t2, numpy.reshape(numpy.exp(lp),(81,81)).T, [0.1], colors='b', linestyles='solid')
  plt.clabel(cs, fontsize=11)

  cs = plt.contour(t1, t2, numpy.reshape(numpy.exp(lp),(81,81)).T, [0.5], colors='k', linestyles='dashed')
  plt.clabel(cs, fontsize=11)

  cs = plt.contour(t1, t2, numpy.reshape(numpy.exp(lp),(81,81)).T, [0.9], colors='r', linestyles='solid')
  plt.clabel(cs, fontsize=11)

  plt.show()
