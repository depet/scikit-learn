"""
=========================================================================
GPML regression: predicting Carbon Dioxide Concentration
=========================================================================

This example is based on the example condacted by Carl Edward Rasmussen, 
available at http://learning.eng.cam.ac.uk/carl/mauna/.

The goal is the model of the CO2 concentration as a function of time x.

The data was downloaded from 
  ftp://ftp.cmdl.noaa.gov/ccg/co2/trends/co2_mm_mlo.txt
and pre-processed using
  tail -n 637 co2_mm_mlo.txt | awk '{ print $3, " ", $4 }' > mauna.txt

A complex covariance function is derived by combining several different 
kinds of simple covariance functions, and the resulting model provides 
an excellent ﬁt to the data.
"""
print(__doc__)

# Author: Dejan Petelin <http://www.linkedin.com/in/dejanpetelin>
# Licence: BSD 3 clause

import numpy

try:
  import matplotlib.pyplot as plt
  matplotlib = True
except Exception, e:
  matplotlib = False

from sklearn import gpml

f = open('mauna.txt')
data = numpy.loadtxt(f)

# get rid of missing data
z = data[:,1] != -99.99

# extract year and CO2 concentration
year = numpy.reshape(data[z,0],(-1,1))
co2 = numpy.reshape(data[z,1],(-1,1))

#training data
x = numpy.reshape(year[year<2004],(-1,1))
y = numpy.reshape(co2[year<2004],(-1,1))

# test data
xx = numpy.reshape(year[year>2004],(-1,1))
yy = numpy.reshape(co2[year>2004],(-1,1))

# covariance function
# covariance contributions, long term trend
k1 = gpml.cov.seIso
# close to periodic component
k2 = (gpml.cov.prod, (gpml.cov.periodic, gpml.cov.seIsoU))
# fluctations with different length-scales
k3 = gpml.cov.rqIso
# very short term (month to month) correlations
k4 = gpml.cov.seIso
# add up covariance terms
covfunc = (gpml.cov.sum, (k1, k2, k3, k4))

hyp = {}
hyp['cov'] = numpy.array([[4, 4, 0, 0, 1, 4, 0, 0, -1, -2, -2]]).T
hyp['lik'] = numpy.array([[-2]])

gp = gpml.GP(x, y, hyp, gpml.inf.exact, gpml.mean.zero, covfunc, gpml.lik.gauss)
gp = gp.fit(x, y, maxiter=500)

# make predictions 20 years into the future
zz = numpy.reshape(numpy.arange(2004+float(1)/24,2024-float(1)/24,float(1)/12),(-1,1))
mu, s2 = gp.predict(zz)

if matplotlib:
  plt.fill_between(numpy.reshape(zz,(-1,)),numpy.reshape(mu+2*numpy.sqrt(s2),(-1,)),numpy.reshape(mu-2*numpy.sqrt(s2),(-1,)),color='k',alpha=.2)
  plt.plot(numpy.reshape(x,(-1,)),numpy.reshape(y,(-1,)),'b')
  plt.plot(numpy.reshape(xx,(-1,)),numpy.reshape(yy,(-1,)),'r')
  plt.xlabel('$year$')
  plt.ylabel('$CO_2$')
  plt.show()
