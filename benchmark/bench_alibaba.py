import quimb.tensor as qtn
import cotengra as ctg
import argparse
import rosnet
import numpy as np
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    'filename', help='Filename of the circuit description', type=str)
parser.add_argument('--swap-split-gate', type=bool, default=False)
parser.add_argument('--minimize',
                    help='Minimization target',
                    type=str,
                    default='flops')
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
fn = args.filename

if args.swap_split_gate:
    gate_opts = {'contract': 'swap-split-gate', 'max_bond': 2}
else:
    gate_opts = {}

circ = qtn.Circuit.from_qasm_file(fn, gate_opts=gate_opts)
psi_f = qtn.MPS_computational_state('0' * (circ.N))
tn = circ.psi & psi_f
output_inds = []

# inplace full simplify and cast to single precision
tn.full_simplify_(output_inds=output_inds)
tn.astype_('complex64')

# NOTE Use 'greedy' optimizer for reproducible results
# opt = ctg.ReusableHyperOptimizer(
#     methods=['kahypar', 'greedy'],
#     max_repeats=128,
#     minimize=args.minimize,
#     score_compression=0.5,  # deliberately make the optimizer try many methods
#     progbar=True,
#     parallel=True,
# )

info = tn.contract(all, optimize='greedy', get='path-info')
print(str(info).encode('utf-8'))
print(str(math.log2(info.largest_intermediate)))

# find optimal cuts
block_shapes = info.shapes
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

    signatures = info.input_subscripts.split(',')
    for i, sign in enumerate(signatures):
        if any(label in ix_sl for label in sign):
            block_shapes[i] = tuple(map(lambda x: 1 if x[1] else info.shapes[i][x[0]], enumerate(
                label in ix_sl for label in sign)))

# move tensors to rosnet and define block_shape
for i, (tensor, bs) in enumerate(zip(tn.tensors, block_shapes)):
    tensor.modify(data=rosnet.array(tensor.data, bs))

start = time.time()
res = tn.contract(all, optimize='greedy', backend='rosnet')
print(f'Amplitude={res.collect()}')
end = time.time()

print(f'Time={end-start}', flush=True)