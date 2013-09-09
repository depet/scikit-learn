import numpy
import numpy.matlib

# sq_dist - a function to compute a matrix of all pairwise squared distances
# between two sets of vectors, stored in the columns of the two matrices, a
# (of size D by n) and b (of size D by m). If only a single argument is given
# or the second matrix is empty, the missing matrix is taken to be identical
# to the first.
def sq_dist(a, b=None):
  D, n = numpy.shape(a)
  
  # The mean is subtracted from the data beforehand to stabilise the
  # computations. This is OK because the squared error is independent of the
  # mean.
  if b is None:
    mu = numpy.array([numpy.mean(a,1)]).T
    a = a - mu
    b = a
    m = n
  else:
    [d, m] = numpy.shape(b)
    
    if d != D:
      raise AttributeError('Error: column lengths must agree.')
    
    mu = (m/float(n+m))*numpy.array([numpy.mean(b,1)]).T + (n/float(n+m))*numpy.array([numpy.mean(a,1)]).T
    a = a - mu
    b = b - mu

  # compute squared distances
  C = numpy.array([numpy.sum(numpy.multiply(b,b),0)])
  C = C - 2*numpy.dot(a.T,b)
  C = C + numpy.array([numpy.sum(numpy.multiply(a, a),0)]).T
  
  # numerical noise can cause C to negative
  C = numpy.maximum(C,0)
  
  return C


# solve_chol - solve linear equations from the Cholesky factorization.
# Solve A*X = B for X, where A is square, symmetric, positive definite. The
# input to the function is R the Cholesky decomposition of A and the matrix B.
# Example: X = solve_chol(chol(A),B);
def solve_chol(L, B):
  if numpy.size(L,0) != numpy.size(L,1) or numpy.size(L,0) != numpy.size(B,0):
    raise AttributeError('Wrong sizes of matrix arguments.')
  return numpy.linalg.solve(L, numpy.linalg.solve(L.T,B))
