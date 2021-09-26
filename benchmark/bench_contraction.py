from timeit import default_timer as timer
import argparse
import numpy as np
from pycompss.api.api import compss_barrier
import rosnet as rn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("mb", type=int)
    parser.add_argument("nb", type=int)
    parser.add_argument("kb", type=int)
    parser.add_argument("threshold", type=int, default=1000)

    args = parser.parse_args()

    m = [2] * int(args.m)
    n = [2] * int(args.n)
    k = [2] * int(args.k)
    mb = [1] * int(args.mb) + [2] * (int(args.m) - int(args.mb))
    nb = [1] * int(args.nb) + [2] * (int(args.n) - int(args.nb))
    kb = [1] * int(args.kb) + [2] * (int(args.k) - int(args.kb))

    mark_start = timer()

    A = rn.ones((*m, *k), blockshape=(*mb, *kb), dtype=np.complex64)
    compss_barrier()
    mark_init_A = timer()

    B = rn.ones((*n, *k), blockshape=(*nb, *kb), dtype=np.complex64)
    compss_barrier()
    mark_init_B = timer()

    axes = ([i + len(m) for i in range(len(k))], [i + len(n) for i in range(len(k))])
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
