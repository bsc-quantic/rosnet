# import cirq
# import cirq.contrib.quimb as ccq
import quimb
import quimb.tensor as qtn
import cotengra as ctg
import rosnet as rn
import argparse
import math
import time
import cmath

parser = argparse.ArgumentParser()
parser.add_argument("n", help="Number of qubits", type=int)
parser.add_argument("--minimize", help="Minimization target", type=str, default="flops")
parser.add_argument(
    "--cut-size",
    help="Maximum number of entries a tensor can have",
    type=int,
    default=None,
)
parser.add_argument(
    "--cut-slices", help="Minimum number of slices to consider", type=int, default=None
)
parser.add_argument(
    "--cut-overhead",
    help="Maximum increase in total number of floating point operations",
    type=float,
    default=None,
)
parser.add_argument(
    "--cut-minimize",
    help="Parameter to minimize on cut selection",
    type=str,
    choices=["flops", "size", "write", "combo", "limit", "compressed"],
    default="flops",
)
parser.add_argument("--cut-temperature", type=float, default=0.01)
parser.add_argument("--optimizer", type=str, default="greedy")

args = parser.parse_args()
n = int(args.n)

circ = qtn.Circuit(N=n)

for i in range(n - 1):
    circ.h(i, gate_round=i)

    for j in range(i + 1, n):
        m = j - i
        circ.cz(j, i, gate_round=i)

zero = quimb.computational_state("0").reshape((2,))
zero_state = [qtn.Tensor(zero, inds=[i]) for i in circ.psi.outer_inds()]
tn = circ.psi & zero_state
tn.astype_("complex64")

# NOTE Use 'greedy' optimizer for reproducible results
if args.optimizer == "greedy":
    opt = "greedy"
elif args.optimizer == "kahypar":
    opt = ctg.ReusableHyperOptimizer(
        methods=["kahypar"],
        max_repeats=128,
        score_compression=0.5,  # deliberately make the optimizer try many methods
        progbar=True,
        minimize=args.minimize,
        parallel=True,
    )
else:
    raise ValueError("Unknown optimizer")

info = tn.contract(all, optimize=opt, get="path-info")
print(str(info).encode("utf-8"))
print(str(math.log2(info.largest_intermediate)))

# find optimal cuts
blockshapes = info.shapes
if args.cut_size or args.cut_slices or args.cut_overhead:
    sf = ctg.SliceFinder(
        info,
        target_size=int(args.cut_size) if args.cut_size else None,
        target_overhead=float(args.cut_overhead) if args.cut_overhead else None,
        target_slices=int(args.cut_slices) if args.cut_slices else None,
        minimize=args.cut_minimize,
        temperature=float(args.cut_temperature),
    )
    ix_sl, cost_sl = sf.search()
    print(cost_sl)

    signatures = info.input_subscripts.split(",")
    for i, sign in enumerate(signatures):
        if any(label in ix_sl for label in sign):
            blockshapes[i] = tuple(
                map(
                    lambda x: 1 if x[1] else info.shapes[i][x[0]],
                    enumerate(label in ix_sl for label in sign),
                )
            )

# move tensors to rosnet and define blockshape
for i, (tensor, bs) in enumerate(zip(tn.tensors, blockshapes)):
    tensor.modify(data=rn.array(tensor.data, blockshape=bs))

start = time.time()
res = tn.contract(all, optimize=opt, backend="rosnet")
print(f"Amplitude={res.collect()}")
end = time.time()

print(f"Time={end-start}", flush=True)
