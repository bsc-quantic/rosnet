from rosnet.tensor import Tensor
import argparse
import numpy as np
import random as rnd


def main():
    parser = argparse.ArgumentParser(description='Tensor.__init__ test')
    parser.add_argument('n', metavar='n', type=int,
                        nargs=1, help='Tensor rank')
    parser.add_argument('m', type=int, nargs=1, help='Block rank')
    parser.add_argument('--samples', dest='num_samples', type=int, nargs=1,
                        default=10, help='#samples')
    args = parser.parse_args()
    n = args.n[0]
    m = args.m[0]
    num_samples = args.num_samples

    shape = [2] * n
    block_shape = [2] * (m) + [1] * (n-m)
    arr = np.random.rand(*shape)

    print("Array generated! Sampling random locations:")
    samples = [tuple(rnd.randint(0, i-1) for i in shape)
               for _ in range(num_samples)]
    for sample in samples:
        print(f"\t{sample} => {arr[sample]}")

    print("Tensor generated! Sampling random locations:")
    t = Tensor.array(arr, block_shape)
    for sample in samples:
        print(f"\t{sample} => {t[sample]}")

    t.sync()


if __name__ == '__main__':
    main()
