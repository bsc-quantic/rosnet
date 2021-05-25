import rosnet
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('m', help='#rows', type=int)
parser.add_argument('n', help='#cols', type=int)
parser.add_argument('mb', help='#rows/block', type=int)
parser.add_argument('nb', help='#cols/block', type=int)

args = parser.parse_args()
m = args.m
n = args.n
mb = args.mb
nb = args.nb

a = rosnet.rand([m, n], [mb, nb])

axes_v = [1]
u, v = rosnet.schmidt(a, axes_v, chi=5)

ar = rosnet.tensordot(u, v, axes=[(1,), (1,)])
print(f'm={m}, n={n}, mb={mb}, nb={nb} => L2 error = %s' %
      np.sqrt(np.sum(np.power(a.collect() - ar.collect(), 2))))
