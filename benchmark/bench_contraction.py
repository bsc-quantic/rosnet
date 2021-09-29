from timeit import default_timer as timer
import argparse
import numpy as np
from pycompss.api.api import compss_barrier
import rosnet as rn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Bm", help="#blocks in the m index", type=int)
    parser.add_argument("Bn", help="#blocks in the n index", type=int)
    parser.add_argument("Bk", help="#blocks in the k index", type=int)
    parser.add_argument("mb", help="block size in the m index", type=int)
    parser.add_argument("nb", help="block size in the n index", type=int)
    parser.add_argument("kb", help="block size in the k index", type=int)
    parser.add_argument("--threshold", type=int, default=1000)

    args = parser.parse_args()

    Bm = int(args.Bm)
    Bn = int(args.Bn)
    Bk = int(args.Bk)
    mb = int(args.mb)
    nb = int(args.nb)
    kb = int(args.kb)

    m = Bm * mb
    n = Bn * nb
    k = Bk * kb

    mark_start = timer()

    A = rn.ones((m, k), blockshape=(mb, kb), dtype=np.complex64)
    compss_barrier()
    print("A generated")
    mark_init_A = timer()

    B = rn.ones((n, k), blockshape=(nb, kb), dtype=np.complex64)
    compss_barrier()
    print("B generated")
    mark_init_B = timer()

    axes = [(1,), (1,)]
    with rn.tuning.configure(threshold_k=args.threshold):
        C = rn.tensordot(A, B, axes)
    compss_barrier()
    mark_contract = timer()

    time_init_A = mark_init_A - mark_start
    time_init_B = mark_init_B - mark_init_A
    time_contract = mark_contract - mark_init_B

    print("time=" + str(time_contract))


if __name__ == "__main__":
    main()
