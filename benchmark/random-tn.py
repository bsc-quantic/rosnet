import opt_einsum as oe
from rosnet import Tensor, tensordot
from pycompss.api.api import compss_wait_on
import cotengra as ctg
import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('n',
                    help='Number of tensors',
                    type=int)
parser.add_argument('reg',
                    help='Regularity / Mean number of indexes each tensor shares')
parser.add_argument('--out',
                    help='Number of output indexes',
                    type=int,
                    default=0)
parser.add_argument('--d-min',
                    help='Minimum dimension size',
                    type=int,
                    default=1)
parser.add_argument('--d-max',
                    help='Maximum dimension size',
                    type=int,
                    default=2)
parser.add_argument('--optimize',
                    help='Optimization algorithm',
                    type=str,
                    default='greedy')
parser.add_argument('--seed',
                    help='Seed',
                    default=None)
parser.add_argument('--cut-size',
                    help='Maximum number of entries a tensor can have',
                    type=int,
                    default=None)
parser.add_argument('--cut-slices',
                    help='Minimum number of slices to consider',
                    type=int,
                    default=None)
parser.add_argument('--cut-overhead',
                    help='Maximum increase in total number of floating point operations',
                    type=float,
                    default=None)
parser.add_argument('--cut-minimize',
                    help='Parameter to minimize on cut selection',
                    type=str, choices=['flops', 'size',
                                       'write', 'combo', 'limit', 'compressed'],
                    default='flops')
parser.add_argument('--cut-temperature',
                    type=float,
                    default=0.01)

args = parser.parse_args()

# generate random tensor network
eq, shapes = oe.helpers.rand_equation(
    n=int(args.n), reg=int(args.reg), n_out=int(args.out), d_min=int(args.d_min), d_max=int(args.d_max), seed=int(args.seed))


# NOTE: oe.contract_path only needs objects with "shape" attr
class FakeTensor(object):
    def __init__(self, shape, block_shape):
        self.shape = shape
        self.block_shape = block_shape


fakes = [FakeTensor(s, s) for s in shapes]

# find contraction path
path, info = oe.contract_path(eq, *fakes, optimize=args.optimize)
print(f'Largest intermediate: {int(math.log2(info.largest_intermediate))}')

# find optimal cuts
if args.cut_size or args.cut_slices or args.cut_overhead:
    sf = ctg.SliceFinder(info, target_size=int(args.cut_size) if args.cut_size else None,
                         target_overhead=float(args.cut_overhead)
                         if args.cut_overhead else None,
                         target_slices=int(args.cut_slices)
                         if args.cut_slices else None,
                         minimize=args.cut_minimize,
                         temperature=float(args.cut_temperature))
    ix_sl, cost_sl = sf.search()
    print(cost_sl)

    signatures = eq.split('->')[0].split(',')
    for i, (sign, tensor) in enumerate(zip(signatures, fakes)):
        if any(label in ix_sl for label in sign):
            block_shape = tuple(map(lambda x: 1 if x[1] else tensor.shape[x[0]], enumerate(
                label in ix_sl for label in sign)))
            fakes[i].block_shape = block_shape

# initialize tensors
tensors = [Tensor.ones(fake.shape, fake.block_shape,
                       dtype=np.float32) for fake in fakes]

# contract tn
a = oe.contract(eq, *tensors, optimize=args.optimize, backend='rosnet')

amplitude = compss_wait_on(a._blocks[()])
print(f'Amplitude: {amplitude}')
